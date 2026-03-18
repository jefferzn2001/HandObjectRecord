"""
Sequential FoundationPose extrinsics calibration.

Cycles through each camera one at a time (VRAM-safe: only one FP estimator
loaded), detects the object via SAM3, and captures T_object_in_cam.

The object defines the world origin for each recording session.

Can be used as:
  - Standalone script:  python fp_extrinsics.py --object cup
  - Importable module:  from fp_extrinsics import calibrate_cameras
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

OT_ROOT = Path(__file__).resolve().parent.parent.parent / "ObjectTracking"


def calibrate_cameras(
    camera_feeds: list[dict],
    object_name: str,
    confidence: float = 0.5,
    n_frames: int = 10,
    est_refine_iter: int = 3,
    track_refine_iter: int = 5,
) -> dict[str, np.ndarray]:
    """
    Sequentially calibrate each camera using FoundationPose.

    For each camera: show its feed, run FP detection/tracking, let the user
    left-click to capture N frames and average, then move to the next camera.

    Args:
        camera_feeds: List of dicts, each with:
            - "name" (str): Camera label (e.g. "zed", "rs_123456")
            - "grab" (callable): Returns (color_bgr, depth_m) or (None, None)
            - "K" (np.ndarray): 3x3 intrinsics matrix
        object_name: Object name for SAM3 text prompt + mesh lookup.
        confidence: SAM3 detection confidence threshold.
        n_frames: Number of frames to average per capture.
        est_refine_iter: FP registration refinement iterations.
        track_refine_iter: FP tracking refinement iterations.

    Returns:
        Dict mapping camera name -> 4x4 T_object_in_cam (extrinsics).
        Only contains entries for cameras that were successfully calibrated.
    """
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    sys.path.insert(0, str(OT_ROOT))
    from utils.tracking_utils import (
        SAM3Worker, build_estimator, draw_tracking_vis, load_mesh,
        set_logging_format, set_seed,
    )

    set_logging_format()
    set_seed(0)

    mesh_path, _ = load_mesh(object_name)
    debug_dir = f"/tmp/fp_calib/{object_name}"
    os.makedirs(debug_dir, exist_ok=True)

    est, mesh, to_origin, bbox = build_estimator(
        mesh_path=mesh_path, debug_dir=debug_dir,
        est_refine_iter=est_refine_iter,
        track_refine_iter=track_refine_iter, debug=0,
    )

    print("[CALIB] Starting SAM3 in subprocess (avoids GL conflict)...", flush=True)
    sam_worker = SAM3Worker(confidence=confidence)

    results: dict[str, np.ndarray] = {}

    for cam_idx, cam in enumerate(camera_feeds):
        cam_name = cam["name"]
        grab_fn = cam["grab"]
        K = cam["K"]

        print(f"\n[CALIB {cam_idx+1}/{len(camera_feeds)}] "
              f"Calibrating '{cam_name}' — point it at the {object_name}")
        print("[CALIB] Left-click when tracking is stable to capture. "
              "Press 'n' to skip this camera.")

        print(f"[CALIB] Warming up {cam_name}...", flush=True)
        for wi in range(5):
            print(f"  warm-up frame {wi}...", end="", flush=True)
            grab_fn()
            print(" ok", flush=True)
        print(f"[CALIB] Warm-up done", flush=True)

        window = f"Calibrate: {cam_name} ({cam_idx+1}/{len(camera_feeds)})"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        initialized = False
        pose: Optional[np.ndarray] = None
        captured: list[np.ndarray] = []
        done = False
        accepted = False
        frame_count = 0
        sam3_attempts = 0
        waiting_for_sam3 = False
        pending_rgb: Optional[np.ndarray] = None
        pending_depth: Optional[np.ndarray] = None

        def on_click(event, x, y, flags, param):
            nonlocal done
            if event == cv2.EVENT_LBUTTONDOWN:
                done = True

        cv2.setMouseCallback(window, on_click)

        print(f"[CALIB] Showing {cam_name} feed — place the {object_name} in view", flush=True)
        print("[CALIB] Controls: click=accept, n=skip, q/Esc=quit, r=reset tracking", flush=True)

        while not accepted:
            color_bgr, depth_m = grab_fn()
            if color_bgr is None:
                continue
            frame_count += 1

            vis = color_bgr.copy()

            # Draw FP pose overlay when tracking
            if initialized and pose is not None:
                vis = draw_tracking_vis(
                    vis, pose, to_origin, bbox, K,
                    initialized, object_name,
                )

            # Check for async SAM3 result
            if waiting_for_sam3:
                ready, mask = sam_worker.poll()
                if ready:
                    waiting_for_sam3 = False
                    if mask is not None and mask.sum() > 100:
                        print(f"[CALIB] SAM3 found mask with {mask.sum()} pixels, "
                              f"running FP register...", flush=True)
                        try:
                            pose = est.register(
                                K=K, rgb=pending_rgb,
                                depth=pending_depth,
                                ob_mask=mask.astype(bool),
                                iteration=est_refine_iter,
                            )
                            initialized = True
                            pos = pose[:3, 3]
                            print(f"[CALIB] {cam_name}: object detected! "
                                  f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})",
                                  flush=True)
                        except Exception as e:
                            print(f"[CALIB] register failed: {e}", flush=True)
                    else:
                        px = mask.sum() if mask is not None else 0
                        print(f"[CALIB] SAM3 attempt {sam3_attempts}: "
                              f"no valid detection (mask_pixels={px}), retrying...",
                              flush=True)
                    pending_rgb = None
                    pending_depth = None

            # Submit SAM3 request (non-blocking)
            if not initialized and not waiting_for_sam3 and frame_count % 15 == 0:
                sam3_attempts += 1
                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                pending_rgb = color_rgb
                pending_depth = depth_m
                sam_worker.submit(color_rgb, object_name)
                waiting_for_sam3 = True
                if sam3_attempts == 1:
                    print(f"[CALIB] Submitted first SAM3 detection "
                          f"(may take a few seconds)...", flush=True)

            # FP tracking on every frame once initialized
            if initialized:
                try:
                    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                    pose = est.track_one(
                        rgb=color_rgb, depth=depth_m, K=K,
                        iteration=track_refine_iter,
                    )
                except Exception:
                    initialized = False
                    pose = None

            # Build status overlay
            if waiting_for_sam3:
                status = f"SAM3 detecting... (attempt {sam3_attempts})"
                status_color = (0, 165, 255)
            elif initialized:
                status = "TRACKING"
                status_color = (0, 255, 0)
            else:
                status = f"waiting (attempts: {sam3_attempts})"
                status_color = (0, 0, 255)

            cv2.putText(vis, f"{cam_name}: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(vis, f"captured: {len(captured)}/{n_frames}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            if len(captured) >= n_frames:
                cv2.putText(vis, "READY — click to accept",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)

            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("n"):
                print(f"[CALIB] Skipping {cam_name}")
                break
            if key in (ord("q"), 27):
                cv2.destroyWindow(window)
                _cleanup_fp(est, sam_worker)
                return results
            if key == ord("r"):
                initialized = False
                pose = None
                captured.clear()
                sam3_attempts = 0
                print("[CALIB] Reset — will re-detect", flush=True)

            # Accumulate frames while tracking
            if initialized and pose is not None and len(captured) < n_frames:
                captured.append(pose.copy())
                if len(captured) == n_frames:
                    avg_pose = np.mean(np.stack(captured), axis=0)
                    pos = avg_pose[:3, 3]
                    print(f"[CALIB] {cam_name}: averaged {n_frames} frames, "
                          f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                    print(f"[CALIB] Click to accept, or wait to re-accumulate")

            if done:
                if len(captured) >= n_frames:
                    avg_pose = np.mean(np.stack(captured), axis=0)
                    results[cam_name] = avg_pose
                    print(f"[CALIB] {cam_name}: accepted!")
                    accepted = True
                else:
                    print(f"[CALIB] Need {n_frames} frames, have {len(captured)}. "
                          "Keep camera steady.")
                    done = False

        cv2.destroyWindow(window)

        # Reset estimator state for next camera (but keep model loaded)
        initialized = False
        pose = None

    _cleanup_fp(est, sam_worker)
    return results


def _cleanup_fp(est, sam_worker):
    """Free GPU memory from FP and shut down SAM3 subprocess."""
    import gc
    try:
        import torch
        sam_worker.shutdown()
        del est
        gc.collect()
        torch.cuda.empty_cache()
        print("[CALIB] Freed FP GPU memory, SAM3 subprocess terminated")
    except ImportError:
        sam_worker.shutdown()
        del est
        gc.collect()


def save_extrinsics(
    output_dir: Path,
    extrinsics: dict[str, np.ndarray],
) -> None:
    """
    Save extrinsics to a directory (one .npy per camera).

    Args:
        output_dir: Directory to write files into.
        extrinsics: Dict mapping camera name -> 4x4 pose.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, pose in extrinsics.items():
        path = output_dir / f"{name}_extrinsics.npy"
        np.save(str(path), pose)
        pos = pose[:3, 3]
        print(f"[CALIB] Saved {path.name}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")


# ── Standalone entry point ──────────────────────────────────────────────────

def main():
    """Run calibration as a standalone script — auto-detects whatever is plugged in."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--n-frames", type=int, default=10)
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).resolve().parent),
                        help="Output directory for extrinsics .npy files")
    args = parser.parse_args()

    pipelines = []
    camera_feeds = []
    zed_handle = None

    # ── RealSense (optional) ────────────────────────────────────────────
    try:
        import pyrealsense2 as rs

        ctx = rs.context()
        rs_serials = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
        print(f"[CALIB] Found {len(rs_serials)} RealSense(s): {rs_serials}")

        for serial in rs_serials:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
            cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            profile = pipe.start(cfg)
            align = rs.align(rs.stream.color)
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = np.array([
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            pipelines.append(pipe)

            def make_grab(p=pipe, a=align, ds=depth_scale):
                def grab():
                    frames = p.wait_for_frames()
                    frames = a.process(frames)
                    df = frames.get_depth_frame()
                    cf = frames.get_color_frame()
                    if not df or not cf:
                        return None, None
                    bgr = cv2.cvtColor(np.asanyarray(cf.get_data()), cv2.COLOR_RGB2BGR)
                    dm = np.asanyarray(df.get_data()).astype(np.float32) * ds
                    dm[(dm < 0.1) | (dm > 3.0)] = 0.0
                    return bgr, dm
                return grab

            camera_feeds.append({
                "name": f"rs_{serial}",
                "grab": make_grab(),
                "K": K,
            })
    except ImportError:
        print("[INFO] pyrealsense2 not installed — skipping RealSense")

    # ── ZED (optional) ──────────────────────────────────────────────────
    try:
        import pyzed.sl as sl

        zed = sl.Camera()
        init_p = sl.InitParameters()
        init_p.camera_resolution = sl.RESOLUTION.HD720
        init_p.camera_fps = 30
        init_p.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        init_p.depth_minimum_distance = 0.04
        init_p.depth_maximum_distance = 2.686
        init_p.coordinate_units = sl.UNIT.METER
        err = zed.open(init_p)
        if err == sl.ERROR_CODE.SUCCESS:
            info = zed.get_camera_information()
            cal = info.camera_configuration.calibration_parameters.left_cam
            K_zed = np.array([
                [cal.fx, 0.0, cal.cx],
                [0.0, cal.fy, cal.cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            zed_rt = sl.RuntimeParameters()
            zed_img = sl.Mat()
            zed_dep = sl.Mat()

            def zed_grab():
                if zed.grab(zed_rt) != sl.ERROR_CODE.SUCCESS:
                    return None, None
                zed.retrieve_image(zed_img, sl.VIEW.LEFT)
                zed.retrieve_measure(zed_dep, sl.MEASURE.DEPTH)
                bgr = cv2.cvtColor(zed_img.get_data(), cv2.COLOR_BGRA2BGR)
                dm = zed_dep.get_data().astype(np.float32)
                valid = np.isfinite(dm)
                dm_clean = np.zeros_like(dm)
                dm_clean[valid] = dm[valid]
                return bgr, dm_clean

            camera_feeds.insert(0, {"name": "zed", "grab": zed_grab, "K": K_zed})
            zed_handle = zed
        else:
            print(f"[INFO] ZED not available: {err}")
    except ImportError:
        print("[INFO] pyzed not installed — skipping ZED")

    if not camera_feeds:
        print("[ERROR] No cameras found")
        return

    print(f"\n[CALIB] Calibrating {len(camera_feeds)} cameras: "
          f"{[c['name'] for c in camera_feeds]}")

    try:
        extrinsics = calibrate_cameras(
            camera_feeds, args.object,
            confidence=args.confidence, n_frames=args.n_frames,
        )
        if extrinsics:
            save_extrinsics(Path(args.out), extrinsics)
            print(f"\n[CALIB] Done — {len(extrinsics)} cameras calibrated")
        else:
            print("\n[CALIB] No cameras calibrated")
    finally:
        for p in pipelines:
            p.stop()
        if zed_handle is not None:
            zed_handle.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
