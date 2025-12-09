"""
Test script for debugging depth image quality and camera intrinsics.
Tests:
1. Depth image edge preservation in 16-bit format
2. Camera intrinsics correctness
3. K matrix rotation when images are rotated 180 degrees
4. ZED intrinsics for 1280x720 resolution
"""

import numpy as np
import cv2
from pathlib import Path
import pyrealsense2 as rs
import pyzed.sl as sl


def test_depth_edge_preservation():
    """Test if 16-bit depth images preserve edges from raw depth."""
    print("=" * 60)
    print("TEST 1: Depth Edge Preservation")
    print("=" * 60)
    
    # Check if we have recorded data
    data_root = Path("data")
    if not data_root.exists():
        print("[WARN] No data directory found. Run record.py first to generate test data.")
        return
    
    # Find most recent sequence
    sequences = sorted(data_root.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sequences:
        print("[WARN] No recorded sequences found.")
        return
    
    seq = sequences[0]
    print(f"[INFO] Testing with sequence: {seq.name}")
    
    rs_depth_dir = seq / "realsense" / "depth"
    if not rs_depth_dir.exists():
        print("[WARN] No RealSense depth directory found.")
        return
    
    # Load first few depth images
    depth_files = sorted(rs_depth_dir.glob("*.png"))[:5]
    if not depth_files:
        print("[WARN] No depth images found.")
        return
    
    print(f"\n[INFO] Analyzing {len(depth_files)} depth images...")
    
    for depth_file in depth_files:
        depth_16bit = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        if depth_16bit is None:
            continue
        
        # Analyze depth statistics
        valid = depth_16bit[depth_16bit > 0]
        if valid.size == 0:
            continue
        
        # Check for edge information (gradient magnitude)
        depth_float = depth_16bit.astype(np.float32)
        grad_x = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Edge pixels (high gradient) - use valid mask from 2D array
        valid_mask = depth_16bit > 0
        if valid_mask.any():
            edge_threshold = np.percentile(gradient_magnitude[valid_mask], 90)
            edge_pixels = np.sum(gradient_magnitude > edge_threshold)
        else:
            edge_threshold = 0
            edge_pixels = 0
        
        print(f"\n  File: {depth_file.name}")
        print(f"    Valid pixels: {valid.size}/{depth_16bit.size} ({100*valid.size/depth_16bit.size:.1f}%)")
        print(f"    Depth range: {valid.min()} - {valid.max()} mm")
        print(f"    Median depth: {np.median(valid):.1f} mm")
        print(f"    Edge pixels (90th percentile): {edge_pixels} ({100*edge_pixels/depth_16bit.size:.2f}%)")
        if valid_mask.any():
            print(f"    Gradient stats: min={gradient_magnitude[valid_mask].min():.2f}, "
                  f"max={gradient_magnitude[valid_mask].max():.2f}, "
                  f"median={np.median(gradient_magnitude[valid_mask]):.2f}")
        else:
            print(f"    Gradient stats: No valid pixels")


def test_camera_intrinsics():
    """Test camera intrinsics and K matrix rotation."""
    print("\n" + "=" * 60)
    print("TEST 2: Camera Intrinsics")
    print("=" * 60)
    
    # RealSense intrinsics
    print("\n[INFO] RealSense Camera:")
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        profile = pipeline.start(config)
        
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        
        print(f"  Resolution: {intrinsics.width} x {intrinsics.height}")
        print(f"  Original K matrix:")
        print(f"    fx={intrinsics.fx:.6f}, fy={intrinsics.fy:.6f}")
        print(f"    cx={intrinsics.ppx:.6f}, cy={intrinsics.ppy:.6f}")
        
        # Rotated K matrix (180 degrees)
        new_cx = intrinsics.width - intrinsics.ppx
        new_cy = intrinsics.height - intrinsics.ppy
        print(f"\n  Rotated K matrix (180 deg):")
        print(f"    fx={intrinsics.fx:.6f}, fy={intrinsics.fy:.6f}")
        print(f"    cx={new_cx:.6f}, cy={new_cy:.6f}")
        
        pipeline.stop()
    except Exception as e:
        print(f"  [ERROR] Failed to get RealSense intrinsics: {e}")
    
    # ZED intrinsics
    print("\n[INFO] ZED Camera:")
    try:
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        
        err = zed.open(init_params)
        if err == sl.ERROR_CODE.SUCCESS:
            info = zed.get_camera_information()
            calib = info.camera_configuration.calibration_parameters.left_cam
            
            print(f"  Resolution: {info.camera_configuration.resolution.width} x "
                  f"{info.camera_configuration.resolution.height}")
            print(f"  Original K matrix:")
            print(f"    fx={calib.fx:.6f}, fy={calib.fy:.6f}")
            print(f"    cx={calib.cx:.6f}, cy={calib.cy:.6f}")
            
            # Rotated K matrix (180 degrees)
            width = info.camera_configuration.resolution.width
            height = info.camera_configuration.resolution.height
            new_cx = width - calib.cx
            new_cy = height - calib.cy
            print(f"\n  Rotated K matrix (180 deg):")
            print(f"    fx={calib.fx:.6f}, fy={calib.fy:.6f}")
            print(f"    cx={new_cx:.6f}, cy={new_cy:.6f}")
            
            zed.close()
        else:
            print(f"  [ERROR] Failed to open ZED: {err}")
    except Exception as e:
        print(f"  [ERROR] Failed to get ZED intrinsics: {e}")


def test_depth_conversion():
    """Test depth conversion from raw to 16-bit - verify edge preservation."""
    print("\n" + "=" * 60)
    print("TEST 3: Depth Conversion Quality & Edge Preservation")
    print("=" * 60)
    
    # Check saved depth images
    data_root = Path("data")
    if not data_root.exists():
        print("[WARN] No data directory found.")
        return
    
    sequences = sorted(data_root.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sequences:
        print("[WARN] No recorded sequences found.")
        return
    
    seq = sequences[0]
    rs_depth_dir = seq / "realsense" / "depth"
    zed_depth_dir = seq / "zed" / "depth"
    
    print(f"\n[INFO] Testing depth images from: {seq.name}")
    
    # Test RealSense depth
    if rs_depth_dir.exists():
        depth_files = sorted(rs_depth_dir.glob("*.png"))[:3]
        print(f"\n[INFO] RealSense depth images ({len(depth_files)} files):")
        
        for depth_file in depth_files:
            depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            
            valid_mask = depth > 0
            if not valid_mask.any():
                continue
            
            print(f"\n  {depth_file.name}:")
            print(f"    Type: {depth.dtype}, Shape: {depth.shape}")
            print(f"    Depth range: {depth[valid_mask].min()} - {depth[valid_mask].max()} mm")
            
            # Check quantization - count unique values
            unique_values = len(np.unique(depth[valid_mask]))
            total_valid = np.sum(valid_mask)
            print(f"    Unique depth values: {unique_values}/{total_valid} "
                  f"({100*unique_values/total_valid:.2f}%)")
            
            # Proper edge detection on 16-bit depth
            # Normalize to 0-255 for Canny, but preserve precision
            depth_normalized = depth.astype(np.float32)
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
            if depth_max > depth_min:
                depth_normalized = ((depth_normalized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth, dtype=np.uint8)
            
            # Edge detection
            edges_canny = cv2.Canny(depth_normalized, 30, 100)
            edge_count_canny = np.sum(edges_canny > 0)
            
            # Gradient-based edge detection (more appropriate for depth)
            depth_float = depth.astype(np.float32)
            grad_x = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Edge threshold based on gradient
            grad_threshold = np.percentile(gradient_mag[valid_mask], 85)
            edge_mask_grad = (gradient_mag > grad_threshold) & valid_mask
            edge_count_grad = np.sum(edge_mask_grad)
            
            print(f"    Edge pixels (Canny on normalized): {edge_count_canny} ({100*edge_count_canny/edges_canny.size:.2f}%)")
            print(f"    Edge pixels (Gradient-based): {edge_count_grad} ({100*edge_count_grad/depth.size:.2f}%)")
            print(f"    Gradient range: {gradient_mag[valid_mask].min():.2f} - {gradient_mag[valid_mask].max():.2f}")
            print(f"    Gradient median: {np.median(gradient_mag[valid_mask]):.2f}")
            
            # Check if edges are meaningful (not just noise)
            edge_gradients = gradient_mag[edge_mask_grad]
            if edge_gradients.size > 0:
                print(f"    Edge gradient stats: min={edge_gradients.min():.2f}, "
                      f"max={edge_gradients.max():.2f}, median={np.median(edge_gradients):.2f}")
    
    # Test ZED depth
    if zed_depth_dir.exists():
        depth_files = sorted(zed_depth_dir.glob("*.png"))[:3]
        print(f"\n[INFO] ZED depth images ({len(depth_files)} files):")
        
        for depth_file in depth_files:
            depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            
            valid_mask = depth > 0
            if not valid_mask.any():
                continue
            
            print(f"\n  {depth_file.name}:")
            print(f"    Type: {depth.dtype}, Shape: {depth.shape}")
            print(f"    Depth range: {depth[valid_mask].min()} - {depth[valid_mask].max()} mm")
            
            # Check quantization
            unique_values = len(np.unique(depth[valid_mask]))
            total_valid = np.sum(valid_mask)
            print(f"    Unique depth values: {unique_values}/{total_valid} "
                  f"({100*unique_values/total_valid:.2f}%)")
            
            # Edge detection
            depth_normalized = depth.astype(np.float32)
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
            if depth_max > depth_min:
                depth_normalized = ((depth_normalized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth, dtype=np.uint8)
            
            edges_canny = cv2.Canny(depth_normalized, 30, 100)
            edge_count_canny = np.sum(edges_canny > 0)
            
            # Gradient-based
            depth_float = depth.astype(np.float32)
            grad_x = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            grad_threshold = np.percentile(gradient_mag[valid_mask], 85)
            edge_mask_grad = (gradient_mag > grad_threshold) & valid_mask
            edge_count_grad = np.sum(edge_mask_grad)
            
            print(f"    Edge pixels (Canny on normalized): {edge_count_canny} ({100*edge_count_canny/edges_canny.size:.2f}%)")
            print(f"    Edge pixels (Gradient-based): {edge_count_grad} ({100*edge_count_grad/depth.size:.2f}%)")
            print(f"    Gradient range: {gradient_mag[valid_mask].min():.2f} - {gradient_mag[valid_mask].max():.2f}")
            print(f"    Gradient median: {np.median(gradient_mag[valid_mask]):.2f}")
            
            edge_gradients = gradient_mag[edge_mask_grad]
            if edge_gradients.size > 0:
                print(f"    Edge gradient stats: min={edge_gradients.min():.2f}, "
                      f"max={edge_gradients.max():.2f}, median={np.median(edge_gradients):.2f}")


def compare_raw_vs_saved():
    """Compare raw depth frames with saved 16-bit images."""
    print("\n" + "=" * 60)
    print("TEST 4: Raw vs Saved Depth Comparison")
    print("=" * 60)
    print("\n[INFO] This requires live camera. Testing with saved data...")
    
    # This would ideally compare live raw frames with saved versions
    # For now, just analyze saved data
    data_root = Path("data")
    if data_root.exists():
        sequences = sorted(data_root.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
        if sequences:
            print(f"[INFO] Most recent sequence: {sequences[0].name}")
            print("  [NOTE] For full comparison, modify this test to capture")
            print("         raw frames during recording and compare with saved versions.")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DEPTH IMAGE AND INTRINSICS TESTING")
    print("=" * 60)
    
    test_depth_edge_preservation()
    test_camera_intrinsics()
    test_depth_conversion()
    compare_raw_vs_saved()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print("\n[INFO] Review output above to verify:")
    print("  1. Depth images preserve edge information")
    print("  2. Camera intrinsics are correct for resolution")
    print("  3. K matrix rotation is applied correctly")
    print("  4. 16-bit depth format maintains quality")


if __name__ == "__main__":
    main()

