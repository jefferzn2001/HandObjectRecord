#!/usr/bin/env python3
"""
Record camera intrinsics from factory calibration.

This script connects to both ZED2i and RealSense D435 cameras,
retrieves their factory intrinsics, and saves them as .npy files
for use in the HaMeR + FoundationPose pipeline.

Output files:
    cam1_intrinsics.npy - ZED2i 3x3 intrinsic matrix
    cam2_intrinsics.npy - RealSense D435 3x3 intrinsic matrix

Usage:
    python record_intrinsic.py

Note: Run this once before calibration. The intrinsics are factory
values and don't change unless you change camera resolution.
"""

import numpy as np
from pathlib import Path

import pyrealsense2 as rs
import pyzed.sl as sl


# Camera resolution settings (must match record.py)
RS_WIDTH = 848
RS_HEIGHT = 480
RS_FPS = 30

Z_WIDTH = 1280
Z_HEIGHT = 720
Z_FPS = 30


def get_realsense_intrinsics() -> tuple:
    """
    Get RealSense D435 factory intrinsics.
    
    Returns:
        tuple: (K_matrix, width, height, fx, fy, cx, cy)
    """
    print("[INFO] Connecting to RealSense D435...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.rgb8, RS_FPS)
    config.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, RS_FPS)
    
    profile = pipeline.start(config)
    
    # Get intrinsics from color stream
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    width = intrinsics.width
    height = intrinsics.height
    
    # RealSense is mounted upside down, so we need to adjust principal point
    # When image is rotated 180 degrees: cx' = width - cx, cy' = height - cy
    cx_rotated = width - cx
    cy_rotated = height - cy
    
    # Build 3x3 intrinsic matrix (with rotation adjustment)
    K = np.array([
        [fx, 0.0, cx_rotated],
        [0.0, fy, cy_rotated],
        [0.0, 0.0, 1.0]
    ])
    
    pipeline.stop()
    
    print(f"[INFO] RealSense intrinsics (original): fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")
    print(f"[INFO] RealSense intrinsics (180° rotated): cx'={cx_rotated:.6f}, cy'={cy_rotated:.6f}")
    print(f"[INFO] RealSense resolution: {width}x{height}")
    
    return K, width, height, fx, fy, cx_rotated, cy_rotated


def get_zed_intrinsics() -> tuple:
    """
    Get ZED2i factory intrinsics.
    
    Returns:
        tuple: (K_matrix, width, height, fx, fy, cx, cy)
    """
    print("[INFO] Connecting to ZED2i...")
    
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = Z_FPS
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {err}")
    
    # Get factory calibration intrinsics for LEFT camera
    info = zed.get_camera_information()
    calib = info.camera_configuration.calibration_parameters.left_cam
    
    fx = calib.fx
    fy = calib.fy
    cx = calib.cx
    cy = calib.cy
    
    # Get actual resolution
    width = info.camera_configuration.resolution.width
    height = info.camera_configuration.resolution.height
    
    # ZED is NOT mounted upside down, so no rotation adjustment needed
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])
    
    zed.close()
    
    print(f"[INFO] ZED intrinsics: fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")
    print(f"[INFO] ZED resolution: {width}x{height}")
    
    if width != Z_WIDTH or height != Z_HEIGHT:
        print(f"[WARN] ZED resolution mismatch! Expected {Z_WIDTH}x{Z_HEIGHT}")
    
    return K, width, height, fx, fy, cx, cy


def main():
    """
    Main function to record and save camera intrinsics.
    """
    output_dir = Path(__file__).parent
    
    print("=" * 60)
    print("CAMERA INTRINSICS RECORDER")
    print("=" * 60)
    print()
    
    # Get ZED2i intrinsics (cam1)
    try:
        zed_K, zed_w, zed_h, zed_fx, zed_fy, zed_cx, zed_cy = get_zed_intrinsics()
        zed_path = output_dir / "cam1_intrinsics.npy"
        np.save(zed_path, zed_K)
        print(f"[SAVED] ZED2i intrinsics -> {zed_path}")
        print(f"        K = \n{zed_K}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to get ZED intrinsics: {e}")
        zed_K = None
    
    # Get RealSense intrinsics (cam2)
    try:
        rs_K, rs_w, rs_h, rs_fx, rs_fy, rs_cx, rs_cy = get_realsense_intrinsics()
        rs_path = output_dir / "cam2_intrinsics.npy"
        np.save(rs_path, rs_K)
        print(f"[SAVED] RealSense intrinsics -> {rs_path}")
        print(f"        K = \n{rs_K}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to get RealSense intrinsics: {e}")
        rs_K = None
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if zed_K is not None:
        print(f"cam1 (ZED2i):     {output_dir / 'cam1_intrinsics.npy'}")
    else:
        print("cam1 (ZED2i):     FAILED")
    
    if rs_K is not None:
        print(f"cam2 (RealSense): {output_dir / 'cam2_intrinsics.npy'}")
    else:
        print("cam2 (RealSense): FAILED")
    
    print()
    print("These intrinsics are ready for HaMeR triangulation pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()

