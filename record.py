"""
Record synchronized RGB from any combination of cameras (ZED + RealSense).

Auto-detects whatever is connected — works with 0-1 ZED and 0-N RealSenses.
Intrinsics are read from each camera's SDK and saved per-episode alongside
the extrinsics from the optional FP calibration step.

Usage:
    python record.py --name cup_grasp --object cup
    python record.py --name cup_grasp

    Episodes are saved in data/ as: cup_grasp_001, cup_grasp_002, ...

Controls:
    Left mouse click : start / stop episode (auto-saves on stop)
    r                : reset FP tracking (re-detect)
    q                : quit
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# Force Qt5 to use software OpenGL so it doesn't conflict with
# nvdiffrast's CUDA/GL context — prevents segfault on cv2.namedWindow.
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np

# Optional SDK imports — skip gracefully if not installed / no device
try:
    import pyrealsense2 as rs
    _HAS_RS = True
except ImportError:
    _HAS_RS = False

try:
    import pyzed.sl as sl
    _HAS_ZED = True
except ImportError:
    _HAS_ZED = False


# ── Config ──────────────────────────────────────────────────────────────────

RS_WIDTH = 848
RS_HEIGHT = 480
RS_FPS = 30

RS_AUTO_EXPOSURE = True

ZED_EXPOSURE = 100
ZED_GAIN = 80
ZED_WHITEBALANCE = 4200

PREVIEW_SCALE = 0.7


# ── Helpers ─────────────────────────────────────────────────────────────────


def _set_opt(sensor, option, value, desc=""):
    try:
        sensor.set_option(option, value)
    except RuntimeError as e:
        print(f"[WARN] {desc}: {e}")


def write_camK(path: Path, K: np.ndarray):
    """Write a 3x3 intrinsics matrix to cam_K.txt."""
    path.write_text(
        f"{K[0,0]} {K[0,1]} {K[0,2]}\n"
        f"{K[1,0]} {K[1,1]} {K[1,2]}\n"
        f"{K[2,0]} {K[2,1]} {K[2,2]}\n"
    )


# ── Camera wrappers ────────────────────────────────────────────────────────


class RSCamera:
    """Wraps a single RealSense camera (color + depth for calib, color-only for recording)."""

    def __init__(self, serial: str):
        self.serial = serial
        self.name = f"rs_{serial}"
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.rgb8, RS_FPS)
        cfg.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, RS_FPS)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Configure sensor
        dev = self.profile.get_device()
        ds = dev.first_depth_sensor()
        _set_opt(ds, rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy, "preset")
        if ds.supports(rs.option.emitter_enabled):
            _set_opt(ds, rs.option.emitter_enabled, 1.0, "emitter")
        if ds.supports(rs.option.laser_power):
            lr = ds.get_option_range(rs.option.laser_power)
            _set_opt(ds, rs.option.laser_power, min(305.0, lr.max), "laser")

        try:
            cs = dev.first_color_sensor()
            if cs.supports(rs.option.enable_auto_exposure):
                _set_opt(cs, rs.option.enable_auto_exposure,
                         1.0 if RS_AUTO_EXPOSURE else 0.0, "auto_exp")
            if cs.supports(rs.option.enable_auto_white_balance):
                _set_opt(cs, rs.option.enable_auto_white_balance, 1.0, "auto_wb")
        except Exception:
            pass

        print(f"[CAM] RealSense {serial} ready ({RS_WIDTH}x{RS_HEIGHT})")

    def grab(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (bgr, depth_m)."""
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        df = frames.get_depth_frame()
        cf = frames.get_color_frame()
        if not df or not cf:
            return None, None
        bgr = cv2.cvtColor(np.asanyarray(cf.get_data()), cv2.COLOR_RGB2BGR)
        dm = np.asanyarray(df.get_data()).astype(np.float32) * self.depth_scale
        dm[(dm < 0.1) | (dm > 3.0)] = 0.0
        return bgr, dm

    def grab_color(self) -> Optional[np.ndarray]:
        """Returns BGR only (faster, for recording)."""
        bgr, _ = self.grab()
        return bgr

    def stop(self):
        self.pipeline.stop()


class ZEDCamera:
    """Wraps the ZED2i (color + depth for calib, color-only for recording)."""

    def __init__(self):
        self.name = "zed"
        self.zed = sl.Camera()
        init_p = sl.InitParameters()
        init_p.camera_resolution = sl.RESOLUTION.HD720
        init_p.camera_fps = 30
        init_p.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        init_p.depth_minimum_distance = 0.04
        init_p.depth_maximum_distance = 2.686
        init_p.coordinate_units = sl.UNIT.METER
        err = self.zed.open(init_p)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")

        info = self.zed.get_camera_information()
        cal = info.camera_configuration.calibration_parameters.left_cam
        self.K = np.array([
            [cal.fx, 0.0, cal.cx],
            [0.0, cal.fy, cal.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        self.w = info.camera_configuration.resolution.width
        self.h = info.camera_configuration.resolution.height

        def safe_set(name, val):
            s = getattr(sl.VIDEO_SETTINGS, name, None)
            if s is not None:
                self.zed.set_camera_settings(s, val)

        safe_set("AEC_AGC", 1)
        safe_set("WHITEBALANCE_AUTO", 1)

        self.runtime = sl.RuntimeParameters()
        self._img = sl.Mat()
        self._dep = sl.Mat()
        print(f"[CAM] ZED2i ready ({self.w}x{self.h})")

    def grab(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (bgr, depth_m)."""
        if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None, None
        self.zed.retrieve_image(self._img, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self._dep, sl.MEASURE.DEPTH)
        bgr = cv2.cvtColor(self._img.get_data(), cv2.COLOR_BGRA2BGR)
        dm = self._dep.get_data().astype(np.float32)
        valid = np.isfinite(dm)
        dm_clean = np.zeros_like(dm)
        dm_clean[valid] = dm[valid]
        return bgr, dm_clean

    def grab_color(self) -> Optional[np.ndarray]:
        if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        self.zed.retrieve_image(self._img, sl.VIEW.LEFT)
        return cv2.cvtColor(self._img.get_data(), cv2.COLOR_BGRA2BGR)

    def stop(self):
        self.zed.close()


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", type=str, required=True,
                        help="Session name (e.g. cup_grasp). "
                             "Episodes: cup_grasp_001, cup_grasp_002, ...")
    parser.add_argument("--object", type=str, default=None,
                        help="Object name for FP calibration (e.g. cup). "
                             "If set, runs calibration at session start.")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="SAM3 detection confidence")
    parser.add_argument("--n-calib-frames", type=int, default=10,
                        help="Frames to average during calibration")
    parser.add_argument("--rs-control", action="store_true",
                        help="Enable RealSense tuning keys")
    args = parser.parse_args()

    root = Path(__file__).parent
    session_name = args.name.strip()
    if not session_name:
        print("[ERROR] --name cannot be empty")
        return

    task_path = root / "data" / session_name
    task_path.mkdir(parents=True, exist_ok=True)

    # ── Discover & init cameras (whatever is connected) ────────────────
    cameras: list = []

    if _HAS_ZED:
        try:
            cameras.append(ZEDCamera())
        except Exception as e:
            print(f"[INFO] ZED not available: {e}")
    else:
        print("[INFO] ZED SDK (pyzed) not installed — skipping ZED")

    if _HAS_RS:
        ctx = rs.context()
        rs_serials = [d.get_info(rs.camera_info.serial_number)
                      for d in ctx.query_devices()]
        print(f"[INFO] Found {len(rs_serials)} RealSense(s): {rs_serials}")
        for serial in rs_serials:
            try:
                cameras.append(RSCamera(serial))
            except Exception as e:
                print(f"[WARN] RS {serial} failed: {e}")
    else:
        print("[INFO] RealSense SDK (pyrealsense2) not installed — skipping RS")

    if not cameras:
        print("[ERROR] No cameras available")
        return

    cam_names = [c.name for c in cameras]
    print(f"[INFO] Active cameras: {cam_names}")

    # ── Load camera config (primary RS for FP overlay) ────────────────
    config_path = root / "camera_config.yaml"
    primary_rs_serial: Optional[str] = None
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            primary_rs_serial = cfg.get("primary_rs")
            if primary_rs_serial:
                print(f"[INFO] Primary RS (FP overlay): {primary_rs_serial}")
        except Exception:
            pass

    # Find the primary RS camera object (if it exists)
    primary_cam = None
    for cam in cameras:
        if isinstance(cam, RSCamera) and cam.serial == primary_rs_serial:
            primary_cam = cam
            break
    if primary_cam is None and any(isinstance(c, RSCamera) for c in cameras):
        # Fall back to first RS if config doesn't match
        primary_cam = next(c for c in cameras if isinstance(c, RSCamera))
        print(f"[INFO] Using first RS as primary: {primary_cam.serial}")

    # ── FP calibration + live tracker (optional) ────────────────────────
    session_extrinsics: dict[str, np.ndarray] = {}
    tracker = None  # live FP tracker for primary RS during recording

    if args.object:
        print(f"\n[INFO] Running FP calibration for '{args.object}'...")
        sys.path.insert(0, str(root / "record" / "cam_calib"))
        from fp_extrinsics import calibrate_cameras

        feeds = []
        for cam in cameras:
            feeds.append({
                "name": cam.name,
                "grab": cam.grab,
                "K": cam.K,
            })

        session_extrinsics = calibrate_cameras(
            feeds, args.object,
            confidence=args.confidence,
            n_frames=args.n_calib_frames,
        )

        if session_extrinsics:
            print(f"\n[INFO] Calibrated {len(session_extrinsics)}/{len(cameras)} cameras")
        else:
            print("[WARN] No cameras calibrated. Continuing without extrinsics.")

        # Keep FP alive on primary RS for live overlay during recording
        if primary_cam is not None:
            print(f"[INFO] Loading live FP tracker on {primary_cam.name}...")
            try:
                ot_root = root / "ObjectTracking"
                sys.path.insert(0, str(ot_root))
                from utils.tracking_utils import (
                    SAM3Worker, build_estimator, draw_tracking_vis,
                    load_mesh, set_logging_format, set_seed,
                )
                set_logging_format()
                set_seed(0)
                mesh_path, _ = load_mesh(args.object)
                import os
                os.makedirs(f"/tmp/fp_debug/{args.object}", exist_ok=True)
                _est, _mesh, _to_origin, _bbox = build_estimator(
                    mesh_path=mesh_path,
                    debug_dir=f"/tmp/fp_debug/{args.object}",
                    est_refine_iter=2, track_refine_iter=5, debug=0,
                )
                _sam_worker = SAM3Worker(confidence=args.confidence)

                tracker = {
                    "est": _est, "sam_worker": _sam_worker,
                    "to_origin": _to_origin, "bbox": _bbox,
                    "draw": draw_tracking_vis,
                    "K": primary_cam.K,
                    "cam_name": primary_cam.name,
                    "object": args.object,
                    "pose": None, "initialized": False,
                }
                print(f"[INFO] Live FP tracker ready on {primary_cam.name}")
            except Exception as e:
                print(f"[WARN] Failed to load live FP tracker: {e}")
                tracker = None
    else:
        print("[INFO] No --object specified, skipping FP calibration")

    # ── Recording state ─────────────────────────────────────────────────
    recording = False
    frame_idx = 0
    temp_root: Optional[Path] = None
    cam_dirs: dict[str, Path] = {}
    episode_poses: list[np.ndarray] = []
    episode_origin: Optional[np.ndarray] = None

    existing = set()
    for d in task_path.iterdir():
        if d.is_dir():
            name = d.name.rstrip("_TEMP")
            try:
                existing.add(int(name))
            except ValueError:
                pass
    current_seq = (max(existing) + 1) if existing else 1
    print(f"\n[INFO] Task: {session_name} | Next episode: {current_seq:03d}")

    # ── Window + mouse ──────────────────────────────────────────────────
    window = "Record (left-click = start/stop)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def start_episode():
        nonlocal recording, frame_idx, temp_root, cam_dirs, current_seq
        nonlocal episode_poses, episode_origin

        dir_name = f"{current_seq:03d}_TEMP"
        temp_root = task_path / dir_name
        temp_root.mkdir(parents=True, exist_ok=True)

        cam_dirs = {}
        for cam in cameras:
            cam_dir = temp_root / cam.name
            (cam_dir / "rgb").mkdir(parents=True, exist_ok=True)
            write_camK(cam_dir / "cam_K.txt", cam.K)
            cam_dirs[cam.name] = cam_dir

        (temp_root / "traj").mkdir(parents=True, exist_ok=True)

        # Save session extrinsics into episode
        if session_extrinsics:
            calib_dir = temp_root / "calib"
            calib_dir.mkdir(parents=True, exist_ok=True)
            for name, pose in session_extrinsics.items():
                np.save(str(calib_dir / f"{name}_extrinsics.npy"), pose)

        frame_idx = 0
        episode_poses = []
        episode_origin = None
        recording = True
        print(f"\n[REC] {session_name}/{current_seq:03d} started")

    def stop_episode():
        nonlocal recording, current_seq, temp_root

        recording = False
        if temp_root is None:
            return

        if frame_idx == 0:
            shutil.rmtree(temp_root, ignore_errors=True)
            print(f"[REC] {session_name}/{current_seq:03d}: 0 frames, discarded")
            temp_root = None
            return

        # Save FP poses if we have them
        if episode_poses and tracker is not None:
            fp_dir = temp_root / "traj" / "FP" / tracker["cam_name"]
            fp_dir.mkdir(parents=True, exist_ok=True)
            poses_arr = np.stack(episode_poses, axis=0)
            np.save(str(fp_dir / "object_poses.npy"), poses_arr)
            np.save(str(fp_dir / "object_trajectory.npy"), poses_arr[:, :3, 3])
            print(f"[REC] Saved {len(episode_poses)} FP poses")

        final_name = f"{current_seq:03d}"
        final_path = task_path / final_name
        if final_path.exists():
            print(f"[WARN] {final_path} exists, keeping TEMP dir")
        else:
            temp_root.rename(final_path)
            print(f"[REC] Saved {session_name}/{final_name} ({frame_idx} frames)")

        current_seq += 1
        temp_root = None
        print(f"[INFO] Ready — click to start {session_name}/{current_seq:03d}")

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not recording:
                start_episode()
            else:
                stop_episode()

    cv2.setMouseCallback(window, mouse_cb)
    print("[INFO] Left-click to start/stop recording. 'q' to quit.")
    if tracker:
        print("[INFO] Press 'r' to reset FP tracking (re-detect)")
    print()

    # ── Main loop ───────────────────────────────────────────────────────
    try:
        while True:
            # Grab all cameras (primary RS gets color+depth for FP)
            frames: dict[str, Optional[np.ndarray]] = {}
            primary_depth: Optional[np.ndarray] = None

            for cam in cameras:
                if tracker and cam.name == tracker["cam_name"]:
                    bgr, dm = cam.grab()
                    frames[cam.name] = bgr
                    primary_depth = dm
                else:
                    frames[cam.name] = cam.grab_color()

            # Live FP tracking on primary RS
            if tracker and frames.get(tracker["cam_name"]) is not None and primary_depth is not None:
                t = tracker
                bgr = frames[t["cam_name"]]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if not t["initialized"]:
                    # Check for async SAM3 result
                    if t.get("_waiting"):
                        ready, mask = t["sam_worker"].poll()
                        if ready:
                            t["_waiting"] = False
                            if mask is not None and mask.sum() > 100:
                                try:
                                    t["pose"] = t["est"].register(
                                        K=t["K"], rgb=t["_pending_rgb"],
                                        depth=t["_pending_depth"],
                                        ob_mask=mask.astype(bool), iteration=2,
                                    )
                                    t["initialized"] = True
                                    print(f"[FP] Detected '{t['object']}' "
                                          f"(mask={mask.sum()} px)")
                                except Exception as e:
                                    print(f"[FP] register failed: {e}")
                            else:
                                px = mask.sum() if mask is not None else 0
                                print(f"[FP] SAM3: no detection (mask={px} px)")
                    elif not t.get("_waiting"):
                        t["sam_worker"].submit(rgb, t["object"])
                        t["_waiting"] = True
                        t["_pending_rgb"] = rgb
                        t["_pending_depth"] = primary_depth
                else:
                    try:
                        t["pose"] = t["est"].track_one(
                            rgb=rgb, depth=primary_depth, K=t["K"], iteration=5,
                        )
                    except Exception:
                        t["initialized"] = False
                        t["pose"] = None

            # Save if recording
            if recording:
                for cam in cameras:
                    bgr = frames.get(cam.name)
                    if bgr is not None and cam.name in cam_dirs:
                        path = cam_dirs[cam.name] / "rgb" / f"{frame_idx:06d}.png"
                        cv2.imwrite(str(path), bgr)

                # Save FP pose relative to episode origin
                if tracker and tracker["pose"] is not None:
                    if episode_origin is None:
                        episode_origin = tracker["pose"].copy()
                    rel = np.linalg.inv(episode_origin) @ tracker["pose"]
                    episode_poses.append(rel.copy())
                elif tracker:
                    episode_poses.append(np.eye(4, dtype=np.float32))

                frame_idx += 1

            # ── Preview: primary RS with FP overlay ─────────────────────
            vis_name = tracker["cam_name"] if tracker else None
            if vis_name is None or frames.get(vis_name) is None:
                # Fall back to first available camera
                for cam in cameras:
                    if frames.get(cam.name) is not None:
                        vis_name = cam.name
                        break

            if vis_name is None or frames.get(vis_name) is None:
                continue

            vis = frames[vis_name].copy()

            # FP overlay
            if tracker and vis_name == tracker["cam_name"]:
                vis = tracker["draw"](
                    vis, tracker["pose"], tracker["to_origin"], tracker["bbox"],
                    tracker["K"], tracker["initialized"], tracker["object"],
                )

            if recording:
                cv2.circle(vis, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(vis, f"REC {session_name}/{current_seq:03d}  f{frame_idx}  [r=reset FP]",
                            (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(vis, f"IDLE | next: {session_name}/{current_seq:03d}  [r=reset FP]",
                            (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)

            vis = cv2.resize(vis, (0, 0), fx=PREVIEW_SCALE, fy=PREVIEW_SCALE,
                             interpolation=cv2.INTER_AREA)
            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                if recording:
                    stop_episode()
                break

            if key == ord("r") and tracker:
                tracker["initialized"] = False
                tracker["pose"] = None
                print("[FP] Reset — will re-detect")

    finally:
        if recording and temp_root is not None:
            if frame_idx > 0:
                stop_episode()
            else:
                shutil.rmtree(temp_root, ignore_errors=True)

        if tracker and "sam_worker" in tracker:
            tracker["sam_worker"].shutdown()
        for cam in cameras:
            cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Cameras closed. Done.")


if __name__ == "__main__":
    main()
