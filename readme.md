FPdata — Multi-Camera Recording + FoundationPose + HaMeR
=========================================================

Pipeline: **record (with live FP) → upload → postprocess (one command) → train**

Recording env: **`objtrack`** (Python 3.10 — pyrealsense2, pyzed, torch, SAM3, FP).
Post-processing env: **`hamer`** (workstation).

Primary RealSense for FP overlay: set in `camera_config.yaml` (`primary_rs`).


## 1) Record (laptop)

```bash
cd ~/Desktop/FPdata && conda activate objtrack

python record.py --name cup_grasp --object cup
```

Episodes are saved in `data/cup_grasp/001`, `data/cup_grasp/002`, ... (task/episode layout).

**Flow:**
1. FP calibration cycles through each camera sequentially (VRAM-safe).
   Left-click to accept each camera and advance.
2. FP stays loaded on the primary RealSense — you see the 3D bbox overlay
   in real time during recording to verify tracking.
3. Left-click = start/stop episode (auto-saves). `r` = reinitialize FP (if object drifts). `q` = quit.

Both hand and object trajectories end up in the **same object-centric frame**
because the extrinsics ARE the FP-estimated T_object_in_cam. No AprilTag
offset — the coordinate origin is the object itself.


## 2) Transfer

Zip episode folders → upload to Google Drive → download on workstation.
Episodes are self-contained (RGB + intrinsics + extrinsics + FP poses).


## 3) Post-process (workstation — one command)

```bash
conda activate hamer

# Single episode:
python record/hamer/postprocess.py --data_path data/cup_grasp/001

# All episodes in a task:
python record/hamer/postprocess.py --data_root data/cup_grasp

# With object mesh for 3D plot:
python record/hamer/postprocess.py --data_path data/cup_grasp/001 --mesh cup

# Skip visualization:
python record/hamer/postprocess.py --data_path data/cup_grasp/001 --no_vis
```

This single script does everything:
1. HaMeR 2D hand detection on each camera
2. RANSAC triangulation → 3D hand keypoints (needs ≥2 cameras)
3. EEF pose + gripper action
4. Verification videos (per-camera + combined side-by-side)
5. 3D matplotlib visualization (hand + object merged)


## Output per episode

```
data/cup_grasp/001/
  rs_045322074026/rgb/*.png, cam_K.txt
  zed/rgb/*.png, cam_K.txt              (if ZED used)
  calib/
    rs_045322074026_extrinsics.npy       # T_object_in_cam (4×4)
    zed_extrinsics.npy
  traj/
    ── from recording (FP) ──
    FP/rs_045322074026/
      object_poses.npy                   # (N, 4, 4) per-frame, relative to ep start
      object_trajectory.npy              # (N, 3) xyz

    ── from postprocess.py (HaMeR) ──
    righthand_3d_keypoints.npy           # (N, 21, 3) all MANO keypoints
    hand_trajectory.npy                  # (N, 6, 3)  wrist + 5 fingertips
    eef_pose.npy                         # (N, 7)     pos + quat_xyzw
    retarget_gripper_action.npy          # (N,)       0=open, 1=closed
    hamer_<cam>.mp4                      # per-camera verification
    hamer_combined.mp4                   # side-by-side
    combined_trajectory_3d.png           # matplotlib 3D (hand + object)
    manipulation_replay.mp4              # frame-by-frame 3D replay
```

**Hand and object trajectories share the same object-centric coordinate frame.**


## File structure

```
FPdata/
  record.py                   Main recording script
  camera_config.yaml          Primary RS serial
  record/
    cam_calib/fp_extrinsics.py FP-based calibration
    hamer/postprocess.py       Single post-processing script
  ObjectTracking/              SAM3 + FoundationPose
    object/<name>/<name>.obj   Object meshes
  data/                        Recorded episodes
```
