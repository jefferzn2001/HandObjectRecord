ManipDepthRecord: Data Recording + FoundationPose + HaMeR + Calibration Guide
================================================================================

This guide walks you through:
- Recording synchronized ZED + RealSense RGB/depth data
- Running FoundationPose for object pose
- Preparing and running HaMeR for hand keypoints
- Visualizing combined trajectories
- AprilTag-based extrinsics calibration (at the end)

Assumptions
-----------
- You have this repo cloned (we’ll refer to it as FPdata).
- ZED2i and RealSense D435 are connected and visible.
- You have a conda environment named `fp_cams` for FPdata.


1) Record Data (FPdata)
-----------------------
Open a terminal and activate your FPdata environment:

```bash
cd <path-to>/ManipDepthRecord   # "FPdata" for my pc
conda activate fp_cams
```

Start the recorder:

```bash
python record.py --name <run_name>
```

Important flags to know:
- `--name <run_name>`: Sets the recording folder name under `data/`. Example: `finalJ1`.
- `--use-timestamp`: Prepend a timestamp to `--name` (e.g., `20251215_123456_finalJ1`).
- `--rs-control`: Enables keyboard controls to tune RealSense exposure/gain/presets live (keys shown in the terminal).
- `--skip-sam`: Skips the optional SAM prompt step after recording the first frames.

Usage tips:
- Press SPACE to start recording. The tool creates folder structure and writes both cameras’ intrinsics to `cam_K.txt` under each camera folder.
- Press SPACE again to stop recording and exit automatically. Press `q` to quit anytime.
- Output goes to `data/<run_name>/` with subfolders for `realsense/`, `zed/`, and rendered depth previews.

This after each run sam2d selects the initial position of mesh but you can modify it so it is done after

2) Run FoundationPose (separate terminal, separate folder)
----------------------------------------------------------
FoundationPose lives outside this repo. Open a new terminal for it :

```bash
cd FoundationPose
conda activate foundationpose
python FP.py --camera <zed|realsense> --mesh <ObjectName> --name <run_name>
```

Notes:
- Use the same `<run_name>` you used while recording in FPdata.
- If you want to add a new object:
  - Create a folder under `FoundationPose/object/<ObjectName>/`
  - Put the `.obj` and `.mtl` files inside, and name them exactly `<ObjectName>.obj` and `<ObjectName>.mtl`
  - Then pass `--mesh <ObjectName>`


3) HaMeR Setup and Usage
------------------------
HaMeR runs separately from FPdata. Create a working folder for HaMeR anywhere you like.

Setup:
1. Clone the HaMeR repository (use the official source you prefer):
   ```bash
   git clone <HaMeR repo URL> hamer
   cd hamer
   ```
2. Create/activate your Python/conda environment following the HaMeR README.
   - CUDA/Torch versions can be tricky depending on your GPU drivers.
   - If you hit CUDA/PyTorch mismatch issues, adjust Torch and CUDA toolkit versions accordingly. If stuck, ask GPT with your exact error and `nvidia-smi` output.
3. Copy the following helper files from FPdata into your HaMeR workflow (paths may vary in your setup):
   - From FPdata:
     - `record/hamer/preprocess.py`
     - `record/hamer/save_video_processed.py`
     - `record/hamer/vitpose_model/` (if present in your project; otherwise get it from your team’s shared location)
   - Place these where you plan to run HaMeR (e.g., a `scripts/` folder in your HaMeR checkout or alongside your data path).

Run HaMeR on a recording:
```bash
# Example: run preprocessing to generate hand keypoints and trajectories
python preprocess.py --data_path <absolute_path_to_FPdata>/data/<run_name>

# Optional: generate preview videos
python save_video_processed.py --data_path <absolute_path_to_FPdata>/data/<run_name>
```

Two-hand support:
- If you need to also use left hand while processing, check HaMeR’s configuration files (model/config YAMLs or script flags depending on your fork) and enable both hands accordingly.

Moving data between machines (optional):
----------------------------------------
Sample commands you can adapt to your environment. Replace placeholders with your hostnames and paths.

- Send a local recording folder to a remote machine:
  ```bash
  scp -r <local_FPdata>/data/<run_name> <user>@<remote_host>:<remote_FPdata>/data/
  ```
- Bring back the minimal results from remote (just trajectories):
  ```bash
  rsync -avz --progress <user>@<remote_host>:<remote_FPdata>/data/<run_name>/traj/ <local_FPdata>/data/<run_name>/traj/
  ```


4) Visualize Combined Results (FPdata)
--------------------------------------
Once FoundationPose and HaMeR outputs exist for `<run_name>`, visualize:

```bash
cd <path-to>/ManipDepthRecord
conda activate fp_cams
python visualize3d.py --name <run_name> [--camera zed|realsense] [--mesh <ObjectName>]
```

Tips:
- Mesh display: if `--mesh` is set, the code samples points from the mesh. The mesh directory is set by `FP_MESH_DIR` in `visualize3d.py`. Update it to your local FoundationPose object folder if needed.
- Handy offsets for minor hand alignment tweaks are available: `--hand_offset_x`, `--hand_offset_y`, `--hand_offset_z`.
- Outputs include a static PNG and a replay video under `data/<run_name>/traj/`.


5) AprilTag Extrinsics Calibration (end)
----------------------------------------
Calibrate camera-to-world transforms using AprilTags. The AprilTag defines the world origin and orientation (see details in `record/cam_calib/april_extrinsics.py`).

Physical setup:
- Print an AprilTag: family `tag36h11`, ID `0`, black square size `160mm` (0.16m).
- Place it approximately at the center of the table, face up, with its X-axis pointing your desired “forward” direction.

Run the calibration tool from FPdata:
```bash
python record/cam_calib/april_extrinsics.py
```
Controls:
- SPACE: capture multiple frames and average for stability
- S: save extrinsics
- Q/ESC: quit

This saves:
- `record/cam_calib/cam1_extrinsics.npy` (ZED)
- `record/cam_calib/cam2_extrinsics.npy` (RealSense)

If you also run on a remote machine, copy these up/down as needed:
```bash
# Send extrinsics to another machine
scp record/cam_calib/cam1_extrinsics.npy record/cam_calib/cam2_extrinsics.npy \
    <user>@<remote_host>:<remote_FPdata>/record/cam_calib/
```

With extrinsics in place, `visualize3d.py` will transform object poses from camera to world automatically.


Quick Reference Commands
------------------------
- Record:
  ```bash
  cd <path-to>/ManipDepthRecord && conda activate fp_cams
  python record.py --name <run_name>
  ```
- FoundationPose (new terminal):
  ```bash
  cd <path-to>/FoundationPose
  python FP.py --camera zed --mesh <ObjectName> --name <run_name>
  ```
- HaMeR (after setup):
  ```bash
  python preprocess.py --data_path <abs>/ManipDepthRecord/data/<run_name>
  python save_video_processed.py --data_path <abs>/ManipDepthRecord/data/<run_name>
  ```
- Visualize:xw
  ```bash
  cd <path-to>/ManipDepthRecord && conda activate fp_cams
  python visualize3d.py --name <run_name> --mesh <ObjectName> --camera zed
  ```
- Calibrate (AprilTag):
  ```bash
  python record/cam_calib/april_extrinsics.py
  ```


Notes & Troubleshooting
-----------------------
- If RealSense visualization looks off, enable `--rs-control` and use the on-screen keys to adjust exposure/gain/presets.
- If FoundationPose can’t find your object mesh, verify the object folder and file naming rules under `FoundationPose/object/`.
- If HaMeR install complains about CUDA/Torch versions, match Torch to your CUDA runtime (check `nvidia-smi`) and reinstall Torch/torchvision accordingly. Ask GPT with your exact error for a quick resolution path.
