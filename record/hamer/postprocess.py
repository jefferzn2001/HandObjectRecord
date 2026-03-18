#!/usr/bin/env python3
"""
Single-script post-processing: HaMeR 2D → 3D triangulation → EEF → video → visualization.

Auto-discovers cameras from the episode folder. All outputs (trajectories +
verification videos + matplotlib 3D plots) go into traj/. Hand and object
trajectories are in the same object-centric frame and are merged for visualization.

Usage:
    conda activate hamer
    python postprocess.py --data_path /path/to/data/cup_grasp/001

    # Batch (all episodes under a task):
    python postprocess.py --data_root /path/to/data/cup_grasp

    # Skip visualization:
    python postprocess.py --data_path ... --no_vis

Output (all in episode/traj/):
    righthand_3d_keypoints.npy   (N, 21, 3)  full MANO keypoints
    hand_trajectory.npy          (N, 6, 3)   wrist + 5 fingertips
    eef_pose.npy                 (N, 7)      pos(3) + quat_xyzw(4)
    retarget_gripper_action.npy  (N,)        0=open, 1=closed
    hamer_<cam>.mp4              per-camera HaMeR overlay video
    hamer_combined.mp4           side-by-side (if ≥2 cameras)
    combined_trajectory_3d.png   matplotlib 3D plot (hand + object)
    manipulation_replay.mp4      frame-by-frame 3D replay (optional)
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message="apex is not installed")
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="Fail to import.*MultiScaleDeformableAttention")
warnings.filterwarnings("ignore", message="You are using a MANO model")

import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from natsort import natsorted
import torch

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
MAX_GRIPPER_WIDTH = 0.08500000089406967  # robotiq 2f-85


# ═══════════════════════════════════════════════════════════════════════
# CAMERA DISCOVERY + CALIBRATION
# ═══════════════════════════════════════════════════════════════════════

def discover_cameras(data_path: Path) -> list[str]:
    """Find camera sub-folders that contain rgb/."""
    return [d.name for d in sorted(data_path.iterdir())
            if d.is_dir() and (d / "rgb").is_dir()]


def load_cam_K(cam_dir: Path) -> np.ndarray:
    k_path = cam_dir / "cam_K.txt"
    if not k_path.exists():
        raise FileNotFoundError(f"Intrinsics not found: {k_path}")
    K = np.loadtxt(str(k_path)).reshape(3, 3)
    return K


def load_episode_extrinsics(data_path: Path, cam_names: list[str]) -> dict[str, np.ndarray]:
    calib_dir = data_path / "calib"
    extr = {}
    for name in cam_names:
        p = calib_dir / f"{name}_extrinsics.npy"
        if p.exists():
            extr[name] = np.load(str(p))
    return extr


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — HAMER 2D
# ═══════════════════════════════════════════════════════════════════════

def run_hamer_reconstruction(cam_images_path: Path, out_folder: Path,
                              skip_if_exists: bool = True):
    if skip_if_exists and out_folder.exists():
        existing = list(out_folder.glob("*.npy"))
        if existing:
            print(f"  Already processed ({len(existing)} files) — skipping")
            return

    import hamer

    hamer_dir = Path(hamer.__file__).parent.parent
    abs_cache_dir = (hamer_dir / "_DATA").resolve()
    import hamer.configs as hamer_configs
    hamer_configs.CACHE_DIR_HAMER = str(abs_cache_dir)

    from hamer.configs import CACHE_DIR_HAMER
    from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
    from hamer.models import HAMER, download_models, load_hamer
    from hamer.utils import recursive_to
    from hamer.utils.renderer import Renderer, cam_crop_to_full
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    from vitpose_model import ViTPoseModel

    checkpoint_path = str(abs_cache_dir / "hamer_ckpts" / "checkpoints" / "hamer.ckpt")
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(checkpoint_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    cpm = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    out_folder.mkdir(parents=True, exist_ok=True)

    img_paths = natsorted(
        [p for p in cam_images_path.glob("*.png") if "depth" not in p.stem]
        + [p for p in cam_images_path.glob("*.jpg") if "depth" not in p.stem]
    )
    total = len(img_paths)
    print(f"  {total} images")
    t0 = time.time()

    for idx, img_path in enumerate(img_paths):
        if idx > 0:
            eta = (time.time() - t0) / idx * (total - idx)
            m, s = divmod(int(eta), 60)
            print(f"\r  [{idx+1}/{total}] ETA {m}m{s}s", end="", flush=True)
        else:
            print(f"\r  [{idx+1}/{total}]", end="", flush=True)

        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            continue

        det_out = detector(img_cv2)
        img_rgb = img_cv2[:, :, ::-1]
        det_inst = det_out["instances"]
        valid = (det_inst.pred_classes == 0) & (det_inst.scores > 0.5)
        pred_bboxes = det_inst.pred_boxes.tensor[valid].cpu().numpy()
        pred_scores = det_inst.scores[valid].cpu().numpy()

        vitposes_out = cpm.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes, is_right = [], []
        for vitposes in vitposes_out:
            keyp = vitposes["keypoints"][-21:]
            mask = keyp[:, 2] > 0.5
            if mask.sum() > 3:
                bboxes.append([keyp[mask, 0].min(), keyp[mask, 1].min(),
                               keyp[mask, 0].max(), keyp[mask, 1].max()])
                is_right.append(1)

        img_fn = img_path.stem
        if not bboxes:
            np.save(out_folder / f"{img_fn}_righthand_keypoints.npy",
                    np.full((21, 2), np.nan))
            continue

        dataset = ViTDetDataset(model_cfg, img_cv2, np.stack(bboxes),
                                np.stack(is_right), rescale_factor=2.0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8,
                                             shuffle=False, num_workers=0)
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                out = model(batch)

            try:
                kp2d = (out["pred_keypoints_2d"][0] * batch["box_size"][0].float()
                        + batch["box_center"][0].float().reshape(1, 2)).cpu().numpy()
            except Exception:
                kp2d = np.full((21, 2), np.nan)

            np.save(out_folder / f"{img_fn}_righthand_keypoints.npy", kp2d)

            # Overlay image for verification video
            input_patch = (batch["img"][0].cpu()
                           * (DEFAULT_STD[:, None, None] / 255)
                           + (DEFAULT_MEAN[:, None, None] / 255))
            input_patch = input_patch.permute(1, 2, 0).numpy()
            reg_img = renderer(
                out["pred_vertices"][0].detach().cpu().numpy(),
                out["pred_cam_t"][0].detach().cpu().numpy(),
                batch["img"][0],
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            cv2.imwrite(
                str(out_folder / f"{img_fn}_overlay.png"),
                255 * np.concatenate([input_patch, reg_img], axis=1)[:, :, ::-1],
            )

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    print(f"\n  Done in {m}m{s}s ({elapsed/max(total,1):.2f}s/img)")


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — TRIANGULATION (RANSAC)
# ═══════════════════════════════════════════════════════════════════════

def triangulate_with_ransac(keypoints_list, intrinsics, extrinsics,
                             ransac_iter=500, reproj_threshold=25.0):
    from sklearn.utils import check_random_state

    projections = []
    for i, T_world_cam in enumerate(extrinsics):
        T_cam_world = np.linalg.inv(T_world_cam)
        projections.append(intrinsics[i] @ T_cam_world[:3, :])

    n_kps = keypoints_list[0].shape[0]
    kp3d = np.zeros((n_kps, 3))
    inlier_counts = np.zeros(n_kps, dtype=int)

    for kp_id in range(n_kps):
        obs = [(ci, keypoints_list[ci][kp_id][0], keypoints_list[ci][kp_id][1])
               for ci in range(len(intrinsics))
               if not np.isnan(keypoints_list[ci][kp_id]).any()]

        if len(obs) < 2:
            kp3d[kp_id] = np.nan
            continue

        best_inliers, best_pt = [], None
        rng = check_random_state(None)

        for _ in range(ransac_iter):
            si = rng.choice(len(obs), size=min(2, len(obs)), replace=False)
            A = []
            for idx in si:
                ci, x, y = obs[idx]
                P = projections[ci]
                A.append(x * P[2] - P[0])
                A.append(y * P[2] - P[1])
            _, _, V = np.linalg.svd(np.array(A))
            cand = V[-1, :3] / V[-1, 3]

            inliers = [o for o in obs
                       if np.linalg.norm((projections[o[0]] @ np.append(cand, 1))[:2]
                                         / (projections[o[0]] @ np.append(cand, 1))[2]
                                         - [o[1], o[2]]) < reproj_threshold]
            if len(inliers) > len(best_inliers):
                best_inliers, best_pt = inliers, cand

        if best_pt is not None and len(best_inliers) >= 2:
            A = []
            for ci, x, y in best_inliers:
                P = projections[ci]
                A.append(y * P[2] - P[1])
                A.append(x * P[2] - P[0])
            _, _, V = np.linalg.svd(np.array(A))
            kp3d[kp_id] = V[-1, :3] / V[-1, 3]
            inlier_counts[kp_id] = len(best_inliers)
        else:
            kp3d[kp_id] = np.nan

    return kp3d, inlier_counts


def triangulate_3d_keypoints(data_path, cam_names, intrinsics, extrinsics):
    traj_dir = data_path / "traj"
    traj_dir.mkdir(parents=True, exist_ok=True)

    first_proc = data_path / cam_names[0] / "processed"
    frames = natsorted([
        f.stem.replace("_righthand_keypoints", "")
        for f in first_proc.glob("*_righthand_keypoints.npy")
    ])
    print(f"  {len(frames)} frames")

    kp3d_all = []
    for frame_id in frames:
        kps = []
        for cn in cam_names:
            p = data_path / cn / "processed" / f"{frame_id}_righthand_keypoints.npy"
            try:
                kps.append(np.load(p))
            except Exception:
                kps.append(np.full((21, 2), np.nan))
        kp3d, _ = triangulate_with_ransac(kps, intrinsics, extrinsics)
        kp3d_all.append(kp3d)

    kp3d_arr = np.array(kp3d_all)
    np.save(traj_dir / "righthand_3d_keypoints.npy", kp3d_arr)
    print(f"  Saved righthand_3d_keypoints.npy  {kp3d_arr.shape}")

    KEY_IDX = [0, 4, 8, 12, 16, 20]
    simp = kp3d_arr[:, KEY_IDX, :]
    np.save(traj_dir / "hand_trajectory.npy", simp)
    print(f"  Saved hand_trajectory.npy  {simp.shape}")

    return kp3d_arr


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — EEF POSE + GRIPPER
# ═══════════════════════════════════════════════════════════════════════

def _ema(data, alpha=0.1):
    out = np.zeros_like(data)
    first = next((i for i in range(len(data)) if not np.isnan(data[i])), 0)
    out[:first + 1] = data[first] if not np.isnan(data[first]) else 0.0
    for t in range(first + 1, len(data)):
        out[t] = out[t - 1] if np.isnan(data[t]) else alpha * data[t] + (1 - alpha) * out[t - 1]
    return out


def _rmat_to_quat(R):
    import scipy.spatial.transform as spt
    if np.isnan(R).any() or np.isinf(R).any():
        return np.array([0., 0., 0., 1.])
    det = np.linalg.det(R)
    if np.isnan(det) or abs(det) < 0.5:
        return np.array([0., 0., 0., 1.])
    try:
        return spt.Rotation.from_matrix(R).as_quat()
    except Exception:
        return np.array([0., 0., 0., 1.])


def compute_eef_pose(data_path, alpha=0.1):
    kp_path = data_path / "traj" / "righthand_3d_keypoints.npy"
    if not kp_path.exists():
        print("  No 3D keypoints — skipping EEF")
        return

    kps = np.load(kp_path)
    smoothed = np.zeros_like(kps)
    for ki in range(21):
        for ci in range(3):
            smoothed[:, ki, ci] = _ema(kps[:, ki, ci], alpha)
    kps = smoothed

    eef_list, grip_list = [], []
    for kp in kps:
        eef = np.zeros(7)
        eef[6] = 1.0

        if np.isnan(kp).any():
            eef_list.append(eef_list[-1].copy() if eef_list else eef)
            grip_list.append(grip_list[-1] if grip_list else 0.5)
            continue

        A, B = kp[0], kp[4]
        C = kp[[8, 12, 16, 20]].mean(0)
        ba, ca = B - A, C - A
        if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(ca) < 1e-6:
            eef_list.append(eef_list[-1].copy() if eef_list else eef)
            grip_list.append(grip_list[-1] if grip_list else 0.5)
            continue

        z = (ca + ba); z /= np.linalg.norm(z) + 1e-8
        x = -np.cross(ca, ba); x /= np.linalg.norm(x) + 1e-8
        y = np.cross(z, x)
        eef[:3] = kp[0]
        eef[3:] = _rmat_to_quat(np.column_stack((x, y, z)))
        eef_list.append(eef)

        gw = np.clip(np.linalg.norm(B - C) / MAX_GRIPPER_WIDTH, 0, 1)
        grip_list.append(1 - gw)

    traj = data_path / "traj"
    np.save(traj / "eef_pose.npy", np.array(eef_list))
    np.save(traj / "retarget_gripper_action.npy", np.array(grip_list))
    v = sum(1 for e in eef_list if e[6] != 1.0 or np.any(e[:3] != 0))
    print(f"  Saved eef_pose.npy + retarget_gripper_action.npy  ({v}/{len(eef_list)} valid)")


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — VERIFICATION VIDEOS
# ═══════════════════════════════════════════════════════════════════════

def make_hamer_video(data_path, cam_name, fps=15):
    proc = data_path / cam_name / "processed"
    if not proc.exists():
        return None

    imgs = natsorted(
        [str(p) for p in proc.glob("*_overlay.png")]
        + [str(p) for p in proc.glob("*_overlay.jpg")]
    )
    if not imgs:
        return None

    out_path = data_path / "traj" / f"hamer_{cam_name}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(imgs[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    for p in imgs:
        f = cv2.imread(p)
        if f is not None:
            writer.write(f)
    writer.release()
    print(f"  {cam_name}: {len(imgs)} frames -> {out_path.name}")
    return out_path


def make_combined_video(data_path, cam_names, fps=15):
    """Side-by-side video from ≥2 per-camera videos."""
    vids = []
    for name in cam_names:
        p = data_path / "traj" / f"hamer_{name}.mp4"
        if p.exists():
            vids.append((name, p))
    if len(vids) < 2:
        return

    caps = [(n, cv2.VideoCapture(str(p))) for n, p in vids]
    dims = {n: (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))) for n, c in caps}
    max_h = max(h for _, h in dims.values())
    sw = {n: int(w * max_h / h) for n, (w, h) in dims.items()}
    total_w = sum(sw.values())

    out_path = data_path / "traj" / "hamer_combined.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (total_w, max_h))
    count = 0
    while True:
        parts, any_ok = [], False
        for name, cap in caps:
            ret, frame = cap.read()
            wn = sw[name]
            if not ret:
                parts.append(np.zeros((max_h, wn, 3), dtype=np.uint8))
            else:
                any_ok = True
                if frame.shape[:2] != (max_h, wn):
                    frame = cv2.resize(frame, (wn, max_h))
                cv2.putText(frame, name, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                parts.append(frame)
        if not any_ok:
            break
        writer.write(np.hstack(parts))
        count += 1

    for _, c in caps:
        c.release()
    writer.release()
    print(f"  Combined: {count} frames -> {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — 3D VISUALIZATION (hand + object merged)
# ═══════════════════════════════════════════════════════════════════════

KP_NAMES = ["Wrist", "Thumb", "Index", "Middle", "Ring", "Pinky"]
KP_COLORS = ["#00FF00", "#FF0000", "#FF8800", "#FFFF00", "#00FFFF", "#FF00FF"]


def _discover_fp_camera(data_path: Path):
    """Find FP camera folder (traj/FP/<cam_name>/)."""
    fp_dir = data_path / "traj" / "FP"
    if not fp_dir.exists():
        return None
    subdirs = [d.name for d in fp_dir.iterdir() if d.is_dir()]
    return subdirs[0] if subdirs else None


def _align_trajectories(hand_traj: np.ndarray, object_traj: np.ndarray,
                        object_poses: np.ndarray | None) -> tuple:
    """Align lengths by trimming to shortest. Returns (hand, obj_pos, obj_poses)."""
    n_h = len(hand_traj)
    n_o = len(object_traj) if object_traj is not None else 0
    n_p = len(object_poses) if object_poses is not None else 0
    n = min(n_h, n_o if n_o > 0 else n_h, n_p if n_p > 0 else n_h)
    hand = hand_traj[:n]
    obj_pos = object_traj[:n] if object_traj is not None else None
    obj_poses = object_poses[:n] if object_poses is not None else None
    return hand, obj_pos, obj_poses


def _create_3d_plot(hand_traj: np.ndarray, object_traj: np.ndarray | None,
                    object_poses: np.ndarray | None, mesh_points: np.ndarray | None,
                    title: str, n_mesh_frames: int = 5):
    """Create matplotlib 3D plot. Both trajectories in object-centric frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _transform_mesh(pts: np.ndarray, pose: np.ndarray) -> np.ndarray:
        ones = np.ones((len(pts), 1))
        return (pose @ np.hstack([pts, ones]).T).T[:, :3]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    valid = ~np.any(np.isnan(hand_traj[:, 0, :]), axis=1)
    hand_v = hand_traj[valid]
    if len(hand_v) == 0:
        plt.close(fig)
        return None

    for kp_idx in range(6):
        t = hand_v[:, kp_idx, :]
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color=KP_COLORS[kp_idx],
                linewidth=1.5, alpha=0.7, label=f"Hand: {KP_NAMES[kp_idx]}")
        ax.scatter(t[0, 0], t[0, 1], t[0, 2], color=KP_COLORS[kp_idx],
                   s=80, marker="o", edgecolors="black")
        ax.scatter(t[-1, 0], t[-1, 1], t[-1, 2], color=KP_COLORS[kp_idx],
                   s=80, marker="s", edgecolors="black")

    if object_traj is not None:
        ax.plot(object_traj[:, 0], object_traj[:, 1], object_traj[:, 2],
                color="#8800FF", linewidth=3, alpha=0.9, label="Object center")
        ax.scatter(object_traj[0, 0], object_traj[0, 1], object_traj[0, 2],
                   color="#8800FF", s=150, marker="o", edgecolors="black")
        ax.scatter(object_traj[-1, 0], object_traj[-1, 1], object_traj[-1, 2],
                   color="#8800FF", s=150, marker="s", edgecolors="black")

    if mesh_points is not None and object_poses is not None:
        n_p = len(object_poses)
        idx = np.linspace(0, n_p - 1, min(n_mesh_frames, n_p), dtype=int)
        for i, fi in enumerate(idx):
            mesh_w = _transform_mesh(mesh_points, object_poses[fi])
            alpha = 0.3 + 0.4 * (i / max(len(idx) - 1, 1))
            ax.scatter(mesh_w[:, 0], mesh_w[:, 1], mesh_w[:, 2],
                       c="#8800FF", s=4, alpha=alpha)

    all_pts = hand_v.reshape(-1, 3)
    if object_traj is not None:
        all_pts = np.vstack([all_pts, object_traj])
    all_pts = all_pts[~np.any(np.isnan(all_pts), axis=1)]
    if len(all_pts) == 0:
        plt.close(fig)
        return None
    mid = all_pts.mean(axis=0)
    r = np.ptp(all_pts, axis=0).max() / 2.0 * 1.1
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _create_replay_video(hand_traj: np.ndarray, object_traj: np.ndarray | None,
                         object_poses: np.ndarray | None, mesh_points: np.ndarray | None,
                         out_path: Path, title: str, fps: int = 15, trail: int = 30):
    """Create frame-by-frame 3D replay video."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tempfile
    import shutil

    def _transform_mesh(pts: np.ndarray, pose: np.ndarray) -> np.ndarray:
        ones = np.ones((len(pts), 1))
        return (pose @ np.hstack([pts, ones]).T).T[:, :3]

    n = len(hand_traj)
    valid = ~np.any(np.isnan(hand_traj.reshape(n, -1)), axis=1)
    all_pts = hand_traj[valid].reshape(-1, 3)
    if object_traj is not None:
        all_pts = np.vstack([all_pts, object_traj])
    all_pts = all_pts[~np.any(np.isnan(all_pts), axis=1)]
    if len(all_pts) == 0:
        return
    mid = all_pts.mean(axis=0)
    r = np.ptp(all_pts, axis=0).max() / 2.0 * 1.2

    tmp = Path(tempfile.mkdtemp())
    for fi in range(n):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        start = max(0, fi - trail)
        for kp_idx in range(6):
            trail_pts = hand_traj[start : fi + 1, kp_idx, :]
            ok = ~np.any(np.isnan(trail_pts), axis=1)
            trail_pts = trail_pts[ok]
            if len(trail_pts) > 1:
                ax.plot(trail_pts[:, 0], trail_pts[:, 1], trail_pts[:, 2],
                        color=KP_COLORS[kp_idx], linewidth=1.5, alpha=0.6)
            cur = hand_traj[fi, kp_idx, :]
            if not np.any(np.isnan(cur)):
                ax.scatter(cur[0], cur[1], cur[2], color=KP_COLORS[kp_idx],
                          s=60, edgecolors="black")
        if object_poses is not None and mesh_points is not None and fi < len(object_poses):
            mesh_w = _transform_mesh(mesh_points, object_poses[fi])
            ax.scatter(mesh_w[:, 0], mesh_w[:, 1], mesh_w[:, 2],
                       c="#8800FF", s=6, alpha=0.6)
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)
        ax.set_title(f"{title} — Frame {fi+1}/{n}")
        ax.view_init(elev=25, azim=45)
        fig.savefig(tmp / f"{fi:06d}.png", dpi=80)
        plt.close(fig)
        if (fi + 1) % 50 == 0:
            print(f"    Replay: {fi+1}/{n} frames")

    imgs = sorted(tmp.glob("*.png"))
    if imgs:
        first = cv2.imread(str(imgs[0]))
        h, w = first.shape[:2]
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for p in imgs:
            out.write(cv2.imread(str(p)))
        out.release()
        print(f"  Replay video -> {out_path.name}")
    shutil.rmtree(tmp, ignore_errors=True)


def _load_mesh_points(mesh_dir: Path, n_points: int = 300) -> np.ndarray | None:
    try:
        import trimesh
    except ImportError:
        return None
    objs = list(mesh_dir.glob("*.obj"))
    if not objs:
        return None
    mesh = trimesh.load(str(objs[0]))
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def run_visualization(data_path: Path, mesh_name: str | None = None,
                     n_mesh_frames: int = 5, fps: int = 15, trail: int = 30,
                     interactive: bool = False) -> bool:
    """Merge hand + object trajectories and create 3D matplotlib plots."""
    hand_path = data_path / "traj" / "hand_trajectory.npy"
    if not hand_path.exists():
        print("  [5/5] No hand_trajectory.npy — skipping visualization")
        return False

    hand_traj = np.load(hand_path)
    fp_cam = _discover_fp_camera(data_path)
    object_traj = None
    object_poses = None
    if fp_cam:
        ot_path = data_path / "traj" / "FP" / fp_cam / "object_trajectory.npy"
        op_path = data_path / "traj" / "FP" / fp_cam / "object_poses.npy"
        if ot_path.exists():
            object_traj = np.load(ot_path)
        if op_path.exists():
            object_poses = np.load(op_path)

    hand_traj, object_traj, object_poses = _align_trajectories(
        hand_traj, object_traj, object_poses
    )

    mesh_points = None
    if mesh_name:
        script_root = Path(__file__).resolve().parent.parent.parent
        mesh_dir = script_root / "ObjectTracking" / "object" / mesh_name
        mesh_points = _load_mesh_points(mesh_dir)

    title = f"Hand + Object — {data_path.name}"
    if fp_cam:
        title += f" (FP: {fp_cam})"

    print(f"\n[5/5] 3D visualization")
    fig = _create_3d_plot(hand_traj, object_traj, object_poses, mesh_points, title, n_mesh_frames)
    if fig is None:
        print("  No valid data for plot")
        return False

    png_path = data_path / "traj" / "combined_trajectory_3d.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    print(f"  Saved {png_path.name}")

    video_path = data_path / "traj" / "manipulation_replay.mp4"
    _create_replay_video(hand_traj, object_traj, object_poses, mesh_points,
                        video_path, title, fps, trail)

    import matplotlib.pyplot as plt
    if interactive:
        plt.show()
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def process_episode(data_path: Path, skip_hamer=False, ema_alpha=0.1,
                    run_vis=True, mesh_name=None, no_interactive=True):
    """Run full pipeline on one episode (HaMeR → triangulate → EEF → videos → 3D vis)."""
    cam_names = discover_cameras(data_path)
    if not cam_names:
        print(f"  No camera folders with rgb/ — skipping")
        return

    print(f"\n{'='*60}")
    print(f"Episode: {data_path.name}")
    print(f"Cameras: {cam_names}")
    print(f"{'='*60}")

    # Load calibration
    intrinsics = {n: load_cam_K(data_path / n) for n in cam_names}
    extr_dict = load_episode_extrinsics(data_path, cam_names)
    cams_w_extr = [n for n in cam_names if n in extr_dict]

    print(f"Intrinsics: {len(intrinsics)} cameras")
    print(f"Extrinsics: {len(cams_w_extr)} cameras ({cams_w_extr})")

    # ── Step 1: HaMeR 2D ──
    print(f"\n[1/5] HaMeR 2D reconstruction")
    if not skip_hamer:
        for cn in cam_names:
            rgb = data_path / cn / "rgb"
            out = data_path / cn / "processed"
            if rgb.exists():
                print(f"  Camera: {cn}")
                run_hamer_reconstruction(rgb, out)
    else:
        print("  Skipped (--skip_hamer)")

    # ── Step 2: Triangulate ──
    print(f"\n[2/5] 3D triangulation")
    if len(cams_w_extr) >= 2:
        tri_K = [intrinsics[n] for n in cams_w_extr]
        tri_E = [extr_dict[n] for n in cams_w_extr]
        triangulate_3d_keypoints(data_path, cams_w_extr, tri_K, tri_E)
    else:
        print(f"  Need ≥2 cameras with extrinsics (have {len(cams_w_extr)}) — skipped")

    # ── Step 3: EEF pose ──
    print(f"\n[3/5] EEF pose + gripper")
    compute_eef_pose(data_path, alpha=ema_alpha)

    # ── Step 4: Verification videos ──
    print(f"\n[4/5] Verification videos")
    for cn in cam_names:
        make_hamer_video(data_path, cn)
    if len(cam_names) >= 2:
        make_combined_video(data_path, cam_names)

    # ── Step 5: 3D visualization ──
    if run_vis:
        run_visualization(data_path, mesh_name=mesh_name,
                         interactive=not no_interactive)
    else:
        print(f"\n[5/5] Visualization skipped (--no_vis)")

    # ── Summary ──
    traj = data_path / "traj"
    print(f"\n{'='*60}")
    print(f"DONE — {data_path.name}")
    print(f"{'='*60}")
    print(f"Outputs in: {traj}")
    for f in sorted(traj.iterdir()):
        if f.is_file():
            sz = f.stat().st_size
            if sz > 1_000_000:
                print(f"  {f.name:40s} {sz/1e6:.1f} MB")
            else:
                print(f"  {f.name:40s} {sz/1e3:.0f} KB")


def _is_episode_folder(path: Path) -> bool:
    """True if path has camera subdirs with rgb/ (e.g. rs_xxx/rgb, zed/rgb)."""
    return any((path / c / "rgb").is_dir() for c in path.iterdir() if c.is_dir())


def _collect_episodes(root: Path) -> list[Path]:
    """Collect episode folders. Supports data/task/001 or data/ with task/001 nested."""
    episodes = []
    for d in root.iterdir():
        if not d.is_dir() or d.name.endswith("_TEMP"):
            continue
        if _is_episode_folder(d):
            episodes.append(d)
        else:
            for sub in d.iterdir():
                if sub.is_dir() and _is_episode_folder(sub):
                    episodes.append(sub)
    return natsorted(episodes, key=lambda x: str(x))

def main():
    parser = argparse.ArgumentParser(
        description="Post-process recorded episodes (HaMeR + triangulation + video + 3D vis)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Single episode folder (e.g. data/cup_grasp/001)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Process all episodes under dir (e.g. data/cup_grasp or data/)")
    parser.add_argument("--skip_hamer", action="store_true",
                        help="Skip HaMeR 2D (reuse existing processed/ folders)")
    parser.add_argument("--ema_alpha", type=float, default=0.1,
                        help="EMA smoothing factor")
    parser.add_argument("--no_vis", action="store_true",
                        help="Skip 3D matplotlib visualization")
    parser.add_argument("--mesh", type=str, default=None,
                        help="Object mesh name for 3D plot (e.g. cup)")
    parser.add_argument("--interactive", action="store_true",
                        help="Show interactive matplotlib window after saving")
    args = parser.parse_args()

    if args.data_path:
        dp = Path(args.data_path)
        if not dp.exists():
            raise FileNotFoundError(dp)
        process_episode(dp, skip_hamer=args.skip_hamer, ema_alpha=args.ema_alpha,
                        run_vis=not args.no_vis, mesh_name=args.mesh,
                        no_interactive=not args.interactive)

    elif args.data_root:
        root = Path(args.data_root)
        found = _collect_episodes(root)
        print(f"Found {len(found)} episodes under {root}")
        for ep in found:
            process_episode(ep, skip_hamer=args.skip_hamer, ema_alpha=args.ema_alpha,
                            run_vis=not args.no_vis, mesh_name=args.mesh,
                            no_interactive=not args.interactive)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python postprocess.py --data_path data/cup_grasp/001")
        print("  python postprocess.py --data_root data/cup_grasp")
        print("  python postprocess.py --data_path data/cup_grasp/001 --mesh cup")


if __name__ == "__main__":
    main()
