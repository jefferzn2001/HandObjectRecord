import os
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from typing import Optional

import pyrealsense2 as rs
import pyzed.sl as sl


# ---------- Configs ----------
# RealSense color/depth resolution (recipe: use 848 x 480 for best calibration)
RS_WIDTH = 848
RS_HEIGHT = 480
RS_FPS = 30
RS_STREAM_FORMAT = rs.format.rgb8

# ZED config: HD720 @ 30Hz, NEURAL_PLUS, 0.2–1.2m
Z_WIDTH = 1280
Z_HEIGHT = 720
Z_FPS = 30

# Preview scale so the window isn’t huge
PREVIEW_SCALE = 0.5  # shrink preview window

# Depth operating ranges (meters). Tight ranges reduce noise for tracking.
RS_DEPTH_MIN_M = 0.04  # sensor min/max (matches rs_depth_tuner)
RS_DEPTH_MAX_M = 1.886
Z_DEPTH_MIN_M = 0.04
Z_DEPTH_MAX_M = 2.686

# Visualization depth range (meters) - separate for RealSense and ZED
RS_VIS_MIN_M = 0.04  # RealSense visualization range (53mm)
RS_VIS_MAX_M = 1.886 # RealSense visualization range (1770mm)
ZED_VIS_MIN_M = 0.04  # ZED visualization range (matches current settings)
ZED_VIS_MAX_M = 2.686

# RealSense tuning parameters (matching rs_depth_tuner.py)
RS_COLOR_EXPOSURE = 160.0  # microseconds
RS_COLOR_GAIN = 35.0
RS_DEPTH_AUTO_EXPOSURE = False
RS_DEPTH_EXPOSURE = 100.0
RS_DEPTH_GAIN = 50.0

# RealSense preset sequence (easy to modify for experimentation)
# Sequence: initial -> frame 10 -> frame 20
RS_PRESET_INITIAL = rs.rs400_visual_preset.default  # Initial preset
RS_PRESET_FRAME_10 = rs.rs400_visual_preset.default  # Preset at frame 10
RS_PRESET_FRAME_20 = rs.rs400_visual_preset.high_accuracy  # Final preset at frame 20
# Options: rs.rs400_visual_preset.default, rs.rs400_visual_preset.high_accuracy, rs.rs400_visual_preset.high_density
RS_ENABLE_FILTERS = False  # keep False to match RS Depth Tuner raw output
RS_DECIMATION_FACTOR = 0.0
RS_SPATIAL_ALPHA = 0.5
RS_SPATIAL_DELTA = 20
RS_SPATIAL_MAGNITUDE = 2
RS_SPATIAL_HOLES_FILL = 1
RS_TEMPORAL_ALPHA = 0.7
RS_TEMPORAL_DELTA = 10
RS_HOLE_FILL = 0  # 0-2 in SDK
RS_APPLY_MEDIAN = False
RS_MEDIAN_KERNEL = 3
RS_DEPTH_EMA_ALPHA = 0.0

# ZED manual camera controls
ZED_EXPOSURE = 14  # SDK units (0-100)
ZED_GAIN = 48      # SDK units (0-100)
ZED_WHITEBALANCE = 4200
ZED_CONFIDENCE = 100
ZED_TEXTURE_CONFIDENCE = 100
ZED_DEPTH_STABILITY = 0 # smoothing during motion (0-100)
ZED_DEPTH_EMA_ALPHA = 0.0


def write_camK_txt(path: Path, fx, fy, cx, cy, width=None, height=None, rotated=False):
    """
    Write camera intrinsics K matrix to file.
    
    Args:
        path: Path to output file
        fx, fy: Focal lengths
        cx, cy: Principal point
        width, height: Image dimensions (required if rotated=True)
        rotated: If True, adjust principal point for 180-degree rotation
    """
    if rotated and width is not None and height is not None:
        # When image is rotated 180 degrees, principal point transforms
        new_cx = width - cx
        new_cy = height - cy
        cx, cy = new_cx, new_cy
    
    path.write_text(
        f"{fx} 0.0 {cx}\n"
        f"0.0 {fy} {cy}\n"
        f"0.0 0.0 1.0\n"
    )


def set_option_safe(sensor, option, value, desc: str):
    try:
        sensor.set_option(option, value)
    except RuntimeError as exc:
        print(f"[WARN] Failed to set {desc}: {exc}")




def configure_zed_camera(zed: sl.Camera):
    def safe_set(setting_name, value):
        setting = getattr(sl.VIDEO_SETTINGS, setting_name, None)
        if setting is None:
            print(f"[WARN] ZED setting {setting_name} not available in this SDK.")
            return
        err = zed.set_camera_settings(setting, value)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[WARN] Failed to set ZED setting {setting_name}: {err}")

    safe_set("AEC_AGC", 0)
    safe_set("EXPOSURE", ZED_EXPOSURE)
    safe_set("GAIN", ZED_GAIN)
    safe_set("WHITEBALANCE_TEMPERATURE", ZED_WHITEBALANCE)
def depth_stats(depth_mm):
    """Return rich stats for debugging (matching rs_depth_tuner format)."""
    if depth_mm is None:
        return None

    valid = depth_mm[depth_mm > 0]
    total = depth_mm.size
    if valid.size == 0:
        return {
            "valid_ratio": 0.0,
            "min": None,
            "median": None,
            "max": None,
            "p5": None,
            "p95": None,
            "total": total,
            "n_valid": 0,
        }

    v = valid.astype(np.float32)
    return {
        "valid_ratio": float(valid.size) / float(total),
        "min": float(v.min()),
        "median": float(np.median(v)),
        "max": float(v.max()),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "total": total,
        "n_valid": int(valid.size),
    }


def depth_to_colormap(depth_mm, vis_min_mm=None, vis_max_mm=None):
    """
    Convert depth to colormap (matching rs_depth_tuner.py).
    
    Args:
        depth_mm: Depth array in millimeters
        vis_min_mm: Minimum visualization depth in mm (defaults to RS_VIS_MIN_M * 1000)
        vis_max_mm: Maximum visualization depth in mm (defaults to RS_VIS_MAX_M * 1000)
    """
    if depth_mm is None:
        return None
    
    if vis_min_mm is None:
        vis_min_mm = RS_VIS_MIN_M * 1000.0
    if vis_max_mm is None:
        vis_max_mm = RS_VIS_MAX_M * 1000.0
    
    valid = depth_mm > 0
    if not valid.any():
        norm = np.zeros_like(depth_mm, dtype=np.uint8)
    else:
        depth_f = depth_mm.astype(np.float32)
        norm = (depth_f - vis_min_mm) / max(1.0, (vis_max_mm - vis_min_mm))
        norm = np.clip(norm, 0.0, 1.0)
        norm = (norm * 255.0).astype(np.uint8)
        norm[~valid] = 0
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[depth_mm == 0] = 0
    return color


def setup_realsense():
    """
    Sets up RealSense camera matching rs_depth_tuner.py settings.
    
    Returns:
        tuple: (pipeline, align, depth_scale, rs_calib, rs_filters, depth_sensor, color_sensor, 
                current_preset, current_laser, exp_range, gain_range, laser_range, emitter_supported)
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, RS_STREAM_FORMAT, RS_FPS)
    config.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, RS_FPS)

    profile = pipeline.start(config)

    # Align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    
    # Depth presets (matching rs_depth_tuner.py)
    preset_names = {
        rs.rs400_visual_preset.high_accuracy: "HIGH_ACCURACY",
        rs.rs400_visual_preset.high_density: "HIGH_DENSITY",
        rs.rs400_visual_preset.default: "DEFAULT",
    }
    current_preset = RS_PRESET_INITIAL

    # Set preset initially (will be set again after visualization starts to fix rendering issue)
    set_option_safe(depth_sensor, rs.option.visual_preset, current_preset, "visual preset")
    print(f"[DEPTH] preset -> {preset_names[current_preset]} (will be refreshed after visualization starts)")
    print(f"[DEPTH] Preset sequence: initial={preset_names[RS_PRESET_INITIAL]}, "
          f"frame10={preset_names[RS_PRESET_FRAME_10]}, frame20={preset_names[RS_PRESET_FRAME_20]}")
    
    # Emitter and laser (matching rs_depth_tuner.py - laser set to 305.0)
    emitter_supported = depth_sensor.supports(rs.option.emitter_enabled)
    laser_supported = depth_sensor.supports(rs.option.laser_power)
    laser_range = depth_sensor.get_option_range(rs.option.laser_power) if laser_supported else None
    current_laser = None
    if laser_supported:
        # Set laser to 305.0 (from user settings)
        current_laser = 305.0
        if laser_range:
            current_laser = float(np.clip(current_laser, laser_range.min, laser_range.max))
        set_option_safe(depth_sensor, rs.option.laser_power, current_laser, "laser_power")
        print(f"[DEPTH] initial laser_power = {current_laser}")
    if emitter_supported:
        set_option_safe(depth_sensor, rs.option.emitter_enabled, 1.0, "emitter_enabled")
        print("[DEPTH] emitter_enabled -> 1")
    
    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_ae_val = 1.0 if RS_DEPTH_AUTO_EXPOSURE else 0.0
        set_option_safe(depth_sensor, rs.option.enable_auto_exposure, depth_ae_val, "depth auto exposure")
    if not RS_DEPTH_AUTO_EXPOSURE:
        if depth_sensor.supports(rs.option.exposure):
            set_option_safe(depth_sensor, rs.option.exposure, RS_DEPTH_EXPOSURE, "depth exposure")
        if depth_sensor.supports(rs.option.gain):
            set_option_safe(depth_sensor, rs.option.gain, RS_DEPTH_GAIN, "depth gain")
    # Get min/max distance ranges for interactive control
    min_dist_range = depth_sensor.get_option_range(rs.option.min_distance) if depth_sensor.supports(rs.option.min_distance) else None
    max_dist_range = depth_sensor.get_option_range(rs.option.max_distance) if depth_sensor.supports(rs.option.max_distance) else None
    
    # Initialize current min/max distance values (matching rs_depth_tuner.py: 0.20 and 1.40)
    current_min_dist = 0.20
    current_max_dist = 1.40
    
    # Set min/max distance (matching rs_depth_tuner.py exactly)
    if depth_sensor.supports(rs.option.min_distance):
        set_option_safe(depth_sensor, rs.option.min_distance, current_min_dist, "min_distance")
    if depth_sensor.supports(rs.option.max_distance):
        set_option_safe(depth_sensor, rs.option.max_distance, current_max_dist, "max_distance")
    if depth_sensor.supports(rs.option.confidence_threshold):
        set_option_safe(depth_sensor, rs.option.confidence_threshold, 3, "confidence threshold")

    try:
        color_sensor = device.first_color_sensor()
    except Exception:
        color_sensor = None
    
    # Get exposure/gain ranges (matching rs_depth_tuner.py)
    exp_range = color_sensor.get_option_range(rs.option.exposure) if color_sensor and color_sensor.supports(rs.option.exposure) else None
    gain_range = color_sensor.get_option_range(rs.option.gain) if color_sensor and color_sensor.supports(rs.option.gain) else None
    
    # Clamp initial values to valid ranges
    global RS_COLOR_EXPOSURE, RS_COLOR_GAIN
    if exp_range:
        RS_COLOR_EXPOSURE = float(np.clip(RS_COLOR_EXPOSURE, exp_range.min, exp_range.max))
    if gain_range:
        RS_COLOR_GAIN = float(np.clip(RS_COLOR_GAIN, gain_range.min, gain_range.max))
    
    if color_sensor is not None:
        # Color: manual exposure / gain (matching rs_depth_tuner.py - no white balance)
        if color_sensor.supports(rs.option.enable_auto_exposure):
            set_option_safe(color_sensor, rs.option.enable_auto_exposure, 0.0, "color_auto_exposure")
        if exp_range:
            set_option_safe(color_sensor, rs.option.exposure, RS_COLOR_EXPOSURE, "color_exposure")
        if gain_range:
            set_option_safe(color_sensor, rs.option.gain, RS_COLOR_GAIN, "color_gain")

    rs_filters = []
    if RS_ENABLE_FILTERS:
        if RS_DECIMATION_FACTOR and RS_DECIMATION_FACTOR > 1:
            decimation = rs.decimation_filter()
            decimation.set_option(rs.option.filter_magnitude, RS_DECIMATION_FACTOR)
            rs_filters.append(decimation)

        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, RS_SPATIAL_MAGNITUDE)
        spatial.set_option(rs.option.filter_smooth_alpha, RS_SPATIAL_ALPHA)
        spatial.set_option(rs.option.filter_smooth_delta, RS_SPATIAL_DELTA)
        spatial.set_option(rs.option.holes_fill, RS_SPATIAL_HOLES_FILL)
        rs_filters.append(spatial)

        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, RS_TEMPORAL_ALPHA)
        temporal.set_option(rs.option.filter_smooth_delta, RS_TEMPORAL_DELTA)
        rs_filters.append(temporal)

        hole_fill = rs.hole_filling_filter()
        hole_fill.set_option(rs.option.holes_fill, RS_HOLE_FILL)
        rs_filters.append(hole_fill)

    depth_scale = depth_sensor.get_depth_scale()  # meters per unit

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    rs_calib = (intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

    return (pipeline, align, depth_scale, rs_calib, rs_filters, depth_sensor, color_sensor,
            current_preset, current_laser, exp_range, gain_range, laser_range, emitter_supported,
            min_dist_range, max_dist_range, current_min_dist, current_max_dist)


def setup_zed():
    """
    Sets up ZED camera and returns camera object, runtime params, and calibration intrinsics.
    
    Returns:
        tuple: (zed, runtime, image, depth, calib_params) where calib_params is (fx, fy, cx, cy)
    """
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = Z_FPS
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params.depth_stabilization = ZED_DEPTH_STABILITY
    init_params.depth_minimum_distance = Z_DEPTH_MIN_M
    init_params.depth_maximum_distance = Z_DEPTH_MAX_M
    init_params.coordinate_units = sl.UNIT.METER

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {err}")

    # Get factory calibration intrinsics for LEFT camera
    info = zed.get_camera_information()
    calib = info.camera_configuration.calibration_parameters.left_cam
    zed_fx = calib.fx
    zed_fy = calib.fy
    zed_cx = calib.cx
    zed_cy = calib.cy
    
    # Verify resolution matches expected 1280x720
    actual_width = info.camera_configuration.resolution.width
    actual_height = info.camera_configuration.resolution.height
    print(f"[INFO] ZED resolution: {actual_width}x{actual_height} (expected {Z_WIDTH}x{Z_HEIGHT})")
    print(f"[INFO] ZED intrinsics (factory): fx={zed_fx:.6f}, fy={zed_fy:.6f}, cx={zed_cx:.6f}, cy={zed_cy:.6f}")
    
    if actual_width != Z_WIDTH or actual_height != Z_HEIGHT:
        print(f"[WARN] ZED resolution mismatch! Intrinsics may be incorrect.")

    configure_zed_camera(zed)

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = ZED_CONFIDENCE
    runtime.texture_confidence_threshold = ZED_TEXTURE_CONFIDENCE
    runtime.enable_fill_mode = False

    image = sl.Mat()
    depth = sl.Mat()

    return zed, runtime, image, depth, (zed_fx, zed_fy, zed_cx, zed_cy)


def get_next_numbered_dir(parent_path: Path, person_id: int, version: int) -> Path:
    """
    Get the next numbered subdirectory in parent_path with format V####PERSON####SEQ########.
    If parent doesn't exist, create it. If it exists, ask if user wants to clear it.
    
    When person and version match, uses the maximum existing SEQ number + 1.
    
    Args:
        parent_path: Parent directory path
        person_id: Person ID (4-digit, e.g., 1 -> 0001)
        version: Version ID (4-digit, e.g., 1 -> 0001)
    
    Returns:
        Path to the next numbered directory (e.g., parent_path/V0001PERSON0001SEQ00000001, ...)
    """
    # Create parent if it doesn't exist
    if not parent_path.exists():
        parent_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created parent directory: {parent_path}")
    else:
        # Check if parent has any directories with this person ID and version
        version_person_prefix = f"V{version:04d}PERSON{person_id:04d}SEQ"
        existing_dirs = [d for d in parent_path.iterdir() 
                        if d.is_dir() and d.name.startswith(version_person_prefix)]
        if existing_dirs:
            # Extract sequence numbers
            existing_seqs = []
            for d in existing_dirs:
                try:
                    # Format: V####PERSON####SEQ########
                    seq_part = d.name[len(version_person_prefix):]
                    seq_num = int(seq_part)
                    existing_seqs.append(seq_num)
                except ValueError:
                    continue
            existing_seqs = sorted(existing_seqs)
            print(f"[INFO] Found existing directories for V{version:04d} PERSON{person_id:04d} in {parent_path}: {existing_seqs}")
            response = input(f"[QUESTION] Do you want to clear directories for V{version:04d} PERSON{person_id:04d} in {parent_path}? (y/n): ").strip().lower()
            if response == 'y':
                for d in existing_dirs:
                    shutil.rmtree(d)
                    print(f"[INFO] Removed {d}")
    
    # Find the next available sequence number for this person and version
    # Simply use max + 1 (no gap filling)
    version_person_prefix = f"V{version:04d}PERSON{person_id:04d}SEQ"
    existing_numbers = set()
    for d in parent_path.iterdir():
        if d.is_dir() and d.name.startswith(version_person_prefix):
            try:
                # Extract sequence number from V####PERSON####SEQ########
                seq_part = d.name[len(version_person_prefix):]
                seq_num = int(seq_part)
                existing_numbers.add(seq_num)
            except ValueError:
                continue
    
    if not existing_numbers:
        next_seq = 1
    else:
        next_seq = max(existing_numbers) + 1
    
    # Format as V####PERSON####SEQ########
    dir_name = f"V{version:04d}PERSON{person_id:04d}SEQ{next_seq:08d}"
    next_dir = parent_path / dir_name
    return next_dir


def main():
    global RS_COLOR_EXPOSURE, RS_COLOR_GAIN
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Parent directory path (e.g., 'data/test_wei_02'). Subdirectories will be named PERSON####SEQ########"
    )
    parser.add_argument(
        "--person",
        type=int,
        required=True,
        help="Person ID (e.g., 1, 2, 3). Will be formatted as 4-digit number (0001, 0002, ...)"
    )
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Version ID (e.g., 1, 2, 3). Will be formatted as 4-digit number (0001, 0002, ...). Used to prevent conflicts when code is modified."
    )
    parser.add_argument(
        "--rs-control",
        action="store_true",
        help="Enable interactive RealSense controls (exposure, gain, preset, laser, emitter).",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    
    # Get parent directory path
    parent_path = Path(args.data_path)
    if not parent_path.is_absolute():
        parent_path = root / parent_path
    
    # Validate person ID and version
    if args.person < 1 or args.person > 9999:
        print(f"[ERROR] Person ID must be between 1 and 9999, got {args.person}")
        return
    if args.version < 1 or args.version > 9999:
        print(f"[ERROR] Version ID must be between 1 and 9999, got {args.version}")
        return

    # Init cameras once
    rs_setup = setup_realsense()
    (rs_pipeline, rs_align, rs_depth_scale, rs_calib, rs_filters, rs_depth_sensor, rs_color_sensor,
     rs_current_preset, rs_current_laser, rs_exp_range, rs_gain_range, rs_laser_range, rs_emitter_supported,
     rs_min_dist_range, rs_max_dist_range, rs_current_min_dist, rs_current_max_dist) = rs_setup
    rs_fx, rs_fy, rs_cx, rs_cy = rs_calib
    zed, zed_runtime, zed_image, zed_depth, zed_calib = setup_zed()
    zed_fx, zed_fy, zed_cx, zed_cy = zed_calib
    
    # RealSense preset names for interactive control
    rs_preset_names = {
        rs.rs400_visual_preset.high_accuracy: "HIGH_ACCURACY",
        rs.rs400_visual_preset.high_density: "HIGH_DENSITY",
        rs.rs400_visual_preset.default: "DEFAULT",
    }
    
    # Enable interactive controls only if flag is set
    rs_interactive_enabled = args.rs_control
    
    if rs_interactive_enabled:
        print("[INFO] RealSense interactive controls ENABLED (matching rs_depth_tuner):")
        print("  e/d : exposure +/-")
        print("  r/f : gain +/-")
        print("  1/2/3 : depth preset (HIGH_ACCURACY / HIGH_DENSITY / DEFAULT)")
        print("  z : toggle emitter on/off")
        print("  c/v : laser power +/-")
        print("  w/x : min distance +/- (meters)")
        print("  i/k : max distance +/- (meters)")
        print("  t : print depth debug stats")
        print("  s : save settings and disable interactive controls")
        if rs_exp_range:
            print(f"[INFO] Exposure range: [{rs_exp_range.min}, {rs_exp_range.max}] (current {RS_COLOR_EXPOSURE})")
        if rs_gain_range:
            print(f"[INFO] Gain range:     [{rs_gain_range.min}, {rs_gain_range.max}] (current {RS_COLOR_GAIN})")
        if rs_laser_range:
            print(f"[INFO] Laser range:    [{rs_laser_range.min}, {rs_laser_range.max}] (current {rs_current_laser})")
        if rs_min_dist_range:
            print(f"[INFO] Min distance range: [{rs_min_dist_range.min:.3f}, {rs_min_dist_range.max:.3f}] m (current {rs_current_min_dist:.3f} m)")
        if rs_max_dist_range:
            print(f"[INFO] Max distance range: [{rs_max_dist_range.min:.3f}, {rs_max_dist_range.max:.3f}] m (current {rs_current_max_dist:.3f} m)")
    else:
        print("[INFO] RealSense interactive controls disabled. Use --rs-control to enable.")
        print("[INFO] Press 't' to print depth debug stats.")

    # Recording state
    recording = False
    frame_idx = 0
    seq_root: Optional[Path] = None
    temp_seq_root: Optional[Path] = None  # Temporary directory during recording
    rs_root = None
    zed_root = None
    rs_render_root = None
    zed_render_root = None
    current_seq = 0  # Current sequence number (will be set on first recording)
    waiting_for_save_decision = False  # Flag to indicate we're waiting for y/n decision
    # zed_depth_ema removed - EMA smoothing disabled
    
    # Setup preview window
    preview_window_name = "FPdata preview: RS (left) / ZED (right)"
    cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
    
    # Only create trackbars if interactive controls are enabled
    if rs_interactive_enabled:
        def on_trackbar(_):
            pass  # Trackbar callback (no-op)
        
        # Create separate trackbars for RealSense and ZED visualization depth ranges
        cv2.createTrackbar("RS_min_depth_mm", preview_window_name, int(RS_VIS_MIN_M * 1000), 3000, on_trackbar)
        cv2.createTrackbar("RS_max_depth_mm", preview_window_name, int(RS_VIS_MAX_M * 1000), 3000, on_trackbar)
        cv2.createTrackbar("ZED_min_depth_mm", preview_window_name, int(ZED_VIS_MIN_M * 1000), 3000, on_trackbar)
        cv2.createTrackbar("ZED_max_depth_mm", preview_window_name, int(ZED_VIS_MAX_M * 1000), 3000, on_trackbar)
    
    # Track frames to fix preset initialization timing
    frame_count = 0
    preset_set_to_default = False
    preset_set_to_high_density = False

    try:
        while True:
            frame_count += 1
            
            # Sequence of preset operations to fix initial rendering issue
            # At 10 frames: set to configured preset
            if not preset_set_to_default and frame_count >= 10:
                rs_current_preset = RS_PRESET_FRAME_10
                set_option_safe(rs_depth_sensor, rs.option.visual_preset, rs_current_preset, "visual preset")
                print(f"[DEPTH] Preset set to {rs_preset_names[rs_current_preset]} after {frame_count} frames")
                preset_set_to_default = True
            
            # At 20 frames: set to final configured preset
            if not preset_set_to_high_density and frame_count >= 20:
                rs_current_preset = RS_PRESET_FRAME_20
                set_option_safe(rs_depth_sensor, rs.option.visual_preset, rs_current_preset, "visual preset")
                import time
                time.sleep(0.05)  # Small delay
                set_option_safe(rs_depth_sensor, rs.option.visual_preset, rs_current_preset, "visual preset")
                print(f"[DEPTH] Preset set to {rs_preset_names[rs_current_preset]} after {frame_count} frames (set twice)")
                preset_set_to_high_density = True
            
            # ==================== RealSense ====================
            rs_frames = rs_pipeline.wait_for_frames()
            rs_aligned = rs_align.process(rs_frames)
            depth_frame = rs_aligned.get_depth_frame()
            color_frame = rs_aligned.get_color_frame()

            if not depth_frame or not color_frame:
                # just skip this iteration if RS has a hiccup
                continue

            if rs_filters:
                filtered = depth_frame
                for filt in rs_filters:
                    filtered = filt.process(filtered)
                depth_frame = filtered.as_depth_frame()

            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())      # uint16, device units

            if depth_raw.shape != color.shape[:2]:
                depth_raw = cv2.resize(
                    depth_raw,
                    (color.shape[1], color.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Convert to meters and clamp to reasonable range (matching rs_depth_tuner.py exactly)
            depth_m = depth_raw.astype(np.float32) * rs_depth_scale
            depth_m[(depth_m < 0.1) | (depth_m > 3.0)] = 0.0
            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

            # BGR for OpenCV (stream already BGR8)
            if RS_STREAM_FORMAT == rs.format.bgr8:
                rs_bgr = color.copy()
            else:
                rs_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            
            # Rotate RealSense images 180 degrees (camera mounted upside down)
            rs_bgr = cv2.rotate(rs_bgr, cv2.ROTATE_180)
            depth_mm = cv2.rotate(depth_mm, cv2.ROTATE_180)

            # ==================== ZED ====================
            zed_ok = (zed.grab(zed_runtime) == sl.ERROR_CODE.SUCCESS)
            zed_bgr = None
            depth_mm_zed = None
            zed_vis_color = None

            if zed_ok:
                zed.retrieve_image(zed_image, sl.VIEW.LEFT)
                zed.retrieve_measure(zed_depth, sl.MEASURE.DEPTH)  # meters

                zed_bgra = zed_image.get_data()                    # H,W,4 BGRA
                zed_bgr = cv2.cvtColor(zed_bgra, cv2.COLOR_BGRA2BGR)

                depth_m_raw = zed_depth.get_data().astype(np.float32)  # meters
                valid_mask = np.isfinite(depth_m_raw)
                depth_m = np.zeros_like(depth_m_raw, dtype=np.float32)
                depth_m[valid_mask] = depth_m_raw[valid_mask]
                depth_m[(depth_m < Z_DEPTH_MIN_M) | (depth_m > Z_DEPTH_MAX_M)] = 0.0

                # ZED EMA smoothing removed per user request
                depth_mm_zed = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
                if RS_APPLY_MEDIAN:
                    depth_mm_zed = cv2.medianBlur(depth_mm_zed, RS_MEDIAN_KERNEL)

            # ==================== Visualization ====================
            # Read visualization range from trackbars (if enabled) or use defaults
            if rs_interactive_enabled:
                # Read RealSense visualization range from trackbars
                rs_vis_min_mm = cv2.getTrackbarPos("RS_min_depth_mm", preview_window_name)
                rs_vis_max_mm = cv2.getTrackbarPos("RS_max_depth_mm", preview_window_name)
                if rs_vis_max_mm <= rs_vis_min_mm:
                    rs_vis_max_mm = rs_vis_min_mm + 1
                
                # Read ZED visualization range from trackbars
                zed_vis_min_mm = cv2.getTrackbarPos("ZED_min_depth_mm", preview_window_name)
                zed_vis_max_mm = cv2.getTrackbarPos("ZED_max_depth_mm", preview_window_name)
                if zed_vis_max_mm <= zed_vis_min_mm:
                    zed_vis_max_mm = zed_vis_min_mm + 1
            else:
                # Use default fixed ranges when trackbars are not available
                rs_vis_min_mm = int(RS_VIS_MIN_M * 1000)
                rs_vis_max_mm = int(RS_VIS_MAX_M * 1000)
                zed_vis_min_mm = int(ZED_VIS_MIN_M * 1000)
                zed_vis_max_mm = int(ZED_VIS_MAX_M * 1000)
            
            # RS depth vis (using RS trackbar values)
            rs_vis_color = depth_to_colormap(depth_mm, rs_vis_min_mm, rs_vis_max_mm)
            
            # ZED depth vis (using ZED trackbar values)
            if zed_ok and zed_bgr is not None and depth_mm_zed is not None:
                zed_vis_color = depth_to_colormap(depth_mm_zed, zed_vis_min_mm, zed_vis_max_mm)
            else:
                zed_vis_color = None

            # ==================== Save if recording ====================
            if recording and rs_root is not None and zed_root is not None:
                # RealSense file paths - always save raw depth
                rs_rgb_path = rs_root / "rgb" / f"{frame_idx:06d}.png"
                rs_depth_path = rs_root / "depth" / f"{frame_idx:06d}.png"
                cv2.imwrite(str(rs_rgb_path), rs_bgr)
                cv2.imwrite(str(rs_depth_path), depth_mm)
                
                # RealSense rendered depth (colormap visualization)
                if rs_vis_color is not None and rs_render_root is not None:
                    rs_render_path = rs_render_root / f"{frame_idx:06d}.png"
                    cv2.imwrite(str(rs_render_path), rs_vis_color)

                # ZED file paths - always save raw depth
                if zed_ok and zed_bgr is not None and depth_mm_zed is not None:
                    # ZED raw RGB + depth
                    zed_rgb_path = zed_root / "rgb" / f"{frame_idx:06d}.png"
                    zed_depth_path = zed_root / "depth" / f"{frame_idx:06d}.png"
                    cv2.imwrite(str(zed_rgb_path), zed_bgr)
                    cv2.imwrite(str(zed_depth_path), depth_mm_zed)
                    
                    # ZED rendered depth (colormap visualization)
                    if zed_vis_color is not None and zed_render_root is not None:
                        zed_render_path = zed_render_root / f"{frame_idx:06d}.png"
                        cv2.imwrite(str(zed_render_path), zed_vis_color)

                frame_idx += 1

            if zed_ok and zed_bgr is not None and zed_vis_color is not None:
                # Resize ZED to match RS for stacking (debug view only)
                rs_h, rs_w = rs_bgr.shape[:2]
                if zed_bgr.shape[:2] != (rs_h, rs_w):
                    zed_bgr = cv2.resize(zed_bgr, (rs_w, rs_h))
                    zed_vis_color = cv2.resize(zed_vis_color, (rs_w, rs_h))

                # Show RGB and depth for both cameras
                rgb_row = np.hstack([rs_bgr, zed_bgr])
                depth_row = np.hstack([rs_vis_color, zed_vis_color])
                vis = np.vstack([rgb_row, depth_row])
            else:
                vis = np.vstack([rs_bgr, rs_vis_color])

            # Add status text
            if waiting_for_save_decision:
                cv2.putText(
                    vis,
                    "Recording stopped. Press Y to save, N to discard",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif recording:
                cv2.putText(
                    vis,
                    "REC",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    vis,
                    "Press SPACE to start recording",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            vis_small = cv2.resize(
                vis,
                (0, 0),
                fx=PREVIEW_SCALE,
                fy=PREVIEW_SCALE,
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow(preview_window_name, vis_small)
            key = cv2.waitKey(1) & 0xFF

            # SPACE = toggle recording
            if key == ord(' '):
                if not recording:
                    # Start recording: create temporary folder
                    # First, determine the next sequence number
                    if current_seq == 0:
                        # First recording: find the next available seq number
                        version_person_prefix = f"V{args.version:04d}PERSON{args.person:04d}SEQ"
                        existing_numbers = set()
                        for d in parent_path.iterdir():
                            if d.is_dir() and d.name.startswith(version_person_prefix):
                                try:
                                    seq_part = d.name[len(version_person_prefix):]
                                    seq_num = int(seq_part)
                                    existing_numbers.add(seq_num)
                                except ValueError:
                                    continue
                        if not existing_numbers:
                            current_seq = 1
                        else:
                            current_seq = max(existing_numbers) + 1
                    
                    # Create temporary directory for this recording session
                    temp_dir_name = f"V{args.version:04d}PERSON{args.person:04d}SEQ{current_seq:08d}_TEMP"
                    temp_seq_root = parent_path / temp_dir_name
                    temp_seq_root.mkdir(parents=True, exist_ok=True)
                    print(f"[INFO] Recording started (temporary): {temp_seq_root}")
                    
                    # Create traj folder for HaMeR output
                    traj_root = temp_seq_root / "traj"

                    rs_root = temp_seq_root / "realsense"
                    zed_root = temp_seq_root / "zed"
                    rs_render_root = temp_seq_root / "real_Depth"
                    zed_render_root = temp_seq_root / "Zed_Depth"

                    for d in [
                        rs_root / "rgb",
                        rs_root / "depth",
                        rs_root / "masks",
                        zed_root / "rgb",
                        zed_root / "depth",
                        zed_root / "masks",
                        rs_render_root,
                        zed_render_root,
                        traj_root,
                    ]:
                        d.mkdir(parents=True, exist_ok=True)

                    # Write camera intrinsics for both cameras (with rotation adjustment)
                    # RealSense: 848x480, rotated 180 degrees
                    write_camK_txt(rs_root / "cam_K.txt", rs_fx, rs_fy, rs_cx, rs_cy, 
                                  width=RS_WIDTH, height=RS_HEIGHT, rotated=True)
                    # ZED: 1280x720, not rotated
                    write_camK_txt(zed_root / "cam_K.txt", zed_fx, zed_fy, zed_cx, zed_cy,
                                  width=Z_WIDTH, height=Z_HEIGHT, rotated=False)

                    frame_idx = 0
                    recording = True
                else:
                    # Stop recording: wait for user decision (Y/N key in window)
                    print(f"\n[INFO] Recording stopped at {frame_idx} frames.")
                    print(f"[INFO] Press Y to save, N to discard (in the preview window)")
                    waiting_for_save_decision = True
                    recording = False
            
            # Handle save decision (Y/N keys) when waiting
            if waiting_for_save_decision:
                if key == ord('y') or key == ord('Y'):
                    # Save: rename temporary directory to final name
                    final_dir_name = f"V{args.version:04d}PERSON{args.person:04d}SEQ{current_seq:08d}"
                    final_seq_root = parent_path / final_dir_name
                    
                    if final_seq_root.exists():
                        print(f"[ERROR] Target directory already exists: {final_seq_root}")
                        print(f"[INFO] Keeping temporary directory: {temp_seq_root}")
                    else:
                        temp_seq_root.rename(final_seq_root)
                        print(f"[INFO] Recording saved to: {final_seq_root}")
                        current_seq += 1  # Increment for next recording only if saved
                    
                    # Reset recording state
                    frame_idx = 0
                    seq_root = None
                    temp_seq_root = None
                    rs_root = None
                    zed_root = None
                    rs_render_root = None
                    zed_render_root = None
                    waiting_for_save_decision = False
                    print(f"[INFO] Ready for next recording. Press SPACE to start.")
                elif key == ord('n') or key == ord('N'):
                    # Don't save: delete temporary directory, keep current_seq unchanged
                    if frame_idx == 0:
                        print(f"[INFO] Recording had 0 frames. Restoring to initial state.")
                    else:
                        print(f"[INFO] Discarding recording...")
                    
                    if temp_seq_root is not None and temp_seq_root.exists():
                        import shutil
                        shutil.rmtree(temp_seq_root)
                        print(f"[INFO] Temporary directory deleted.")
                    
                    # Reset recording state (but keep current_seq for next attempt)
                    frame_idx = 0
                    seq_root = None
                    temp_seq_root = None
                    rs_root = None
                    zed_root = None
                    rs_render_root = None
                    zed_render_root = None
                    waiting_for_save_decision = False
                    print(f"[INFO] Ready for next recording. Press SPACE to start.")

            # T = print depth debug stats (matching rs_depth_tuner format)
            if key == ord('t'):
                rs_stats = depth_stats(depth_mm)
                if rs_stats:
                    print("========== DEPTH DEBUG ==========")
                    print(f"Exposure: {RS_COLOR_EXPOSURE} | Gain: {RS_COLOR_GAIN}")
                    if rs_laser_range:
                        print(f"Laser: {rs_current_laser}")
                    if rs_emitter_supported:
                        try:
                            cur_em = rs_depth_sensor.get_option(rs.option.emitter_enabled)
                        except Exception:
                            cur_em = None
                        print(f"Emitter: {cur_em}")
                    if rs_min_dist_range:
                        print(f"Min distance: {rs_current_min_dist:.3f} m")
                    if rs_max_dist_range:
                        print(f"Max distance: {rs_current_max_dist:.3f} m")
                    print(f"Valid pixels: {rs_stats['n_valid']}/{rs_stats['total']} "
                          f"({rs_stats['valid_ratio']*100:.1f}%)")
                    if rs_stats["min"] is None:
                        print("No valid depth.")
                    else:
                        print(f"Depth mm: min={rs_stats['min']:.1f}, median={rs_stats['median']:.1f}, "
                              f"max={rs_stats['max']:.1f}")
                        print(f"         p5={rs_stats['p5']:.1f}, p95={rs_stats['p95']:.1f}")
                    print("=================================")
                else:
                    print("[TEST] No valid depth samples this frame.")
            
            # Save settings and disable interactive controls (s) - only if controls are enabled
            if key == ord('s') and rs_interactive_enabled:
                # Get current emitter state
                emitter_state = None
                if rs_emitter_supported:
                    try:
                        emitter_state = rs_depth_sensor.get_option(rs.option.emitter_enabled)
                    except Exception:
                        emitter_state = 1.0
                
                print("========== SAVED REAL SENSE SETTINGS ==========")
                print(f"Exposure: {RS_COLOR_EXPOSURE}")
                print(f"Gain: {RS_COLOR_GAIN}")
                print(f"Preset: {rs_preset_names[rs_current_preset]}")
                if rs_laser_range and rs_current_laser is not None:
                    print(f"Laser power: {rs_current_laser}")
                if emitter_state is not None:
                    print(f"Emitter enabled: {emitter_state}")
                if rs_min_dist_range:
                    print(f"Min distance: {rs_current_min_dist:.3f} m")
                if rs_max_dist_range:
                    print(f"Max distance: {rs_current_max_dist:.3f} m")
                print("==============================================")
                print("[INFO] RealSense interactive controls disabled.")
                rs_interactive_enabled = False
            
            # RealSense interactive controls (matching rs_depth_tuner.py)
            # Only process if interactive controls are enabled
            if not rs_interactive_enabled:
                pass  # Skip all interactive controls
            else:
                # Exposure up (e), down (d)
                if key == ord('e') and rs_exp_range and rs_color_sensor:
                    RS_COLOR_EXPOSURE = float(np.clip(RS_COLOR_EXPOSURE + 10, rs_exp_range.min, rs_exp_range.max))
                    set_option_safe(rs_color_sensor, rs.option.exposure, RS_COLOR_EXPOSURE, "color_exposure")
                    print(f"[COLOR] exposure = {RS_COLOR_EXPOSURE}")
                if key == ord('d') and rs_exp_range and rs_color_sensor:
                    RS_COLOR_EXPOSURE = float(np.clip(RS_COLOR_EXPOSURE - 10, rs_exp_range.min, rs_exp_range.max))
                    set_option_safe(rs_color_sensor, rs.option.exposure, RS_COLOR_EXPOSURE, "color_exposure")
                    print(f"[COLOR] exposure = {RS_COLOR_EXPOSURE}")
                
                # Gain up (r), down (f)
                if key == ord('r') and rs_gain_range and rs_color_sensor:
                    RS_COLOR_GAIN = float(np.clip(RS_COLOR_GAIN + 5, rs_gain_range.min, rs_gain_range.max))
                    set_option_safe(rs_color_sensor, rs.option.gain, RS_COLOR_GAIN, "color_gain")
                    print(f"[COLOR] gain = {RS_COLOR_GAIN}")
                if key == ord('f') and rs_gain_range and rs_color_sensor:
                    RS_COLOR_GAIN = float(np.clip(RS_COLOR_GAIN - 5, rs_gain_range.min, rs_gain_range.max))
                    set_option_safe(rs_color_sensor, rs.option.gain, RS_COLOR_GAIN, "color_gain")
                    print(f"[COLOR] gain = {RS_COLOR_GAIN}")
                
                # Switch depth presets (1/2/3)
                if key == ord('1'):
                    rs_current_preset = rs.rs400_visual_preset.high_accuracy
                    set_option_safe(rs_depth_sensor, rs.option.visual_preset, rs_current_preset, "visual_preset")
                    print(f"[DEPTH] preset -> {rs_preset_names[rs_current_preset]}")
                if key == ord('2'):
                    rs_current_preset = rs.rs400_visual_preset.high_density
                    set_option_safe(rs_depth_sensor, rs.option.visual_preset, rs_current_preset, "visual_preset")
                    print(f"[DEPTH] preset -> {rs_preset_names[rs_current_preset]}")
                if key == ord('3'):
                    rs_current_preset = rs.rs400_visual_preset.default
                    set_option_safe(rs_depth_sensor, rs.option.visual_preset, rs_current_preset, "visual_preset")
                    print(f"[DEPTH] preset -> {rs_preset_names[rs_current_preset]}")
                
                # Toggle emitter (z)
                if key == ord('z') and rs_emitter_supported:
                    try:
                        cur = rs_depth_sensor.get_option(rs.option.emitter_enabled)
                    except Exception:
                        cur = 1.0
                    new_val = 0.0 if cur > 0.5 else 1.0
                    set_option_safe(rs_depth_sensor, rs.option.emitter_enabled, new_val, "emitter_enabled")
                    print(f"[DEPTH] emitter_enabled -> {new_val}")
                
                # Laser power up/down (c/v)
                if key == ord('c') and rs_laser_range and rs_current_laser is not None:
                    rs_current_laser = float(np.clip(rs_current_laser + 5.0, rs_laser_range.min, rs_laser_range.max))
                    set_option_safe(rs_depth_sensor, rs.option.laser_power, rs_current_laser, "laser_power")
                    print(f"[DEPTH] laser_power = {rs_current_laser}")
                if key == ord('v') and rs_laser_range and rs_current_laser is not None:
                    rs_current_laser = float(np.clip(rs_current_laser - 5.0, rs_laser_range.min, rs_laser_range.max))
                    set_option_safe(rs_depth_sensor, rs.option.laser_power, rs_current_laser, "laser_power")
                    print(f"[DEPTH] laser_power = {rs_current_laser}")
                
                # Min distance up/down (w/x)
                if key == ord('w') and rs_min_dist_range:
                    rs_current_min_dist = float(np.clip(rs_current_min_dist + 0.01, rs_min_dist_range.min, rs_min_dist_range.max))
                    set_option_safe(rs_depth_sensor, rs.option.min_distance, rs_current_min_dist, "min_distance")
                    print(f"[DEPTH] min_distance = {rs_current_min_dist:.3f} m")
                if key == ord('x') and rs_min_dist_range:
                    rs_current_min_dist = float(np.clip(rs_current_min_dist - 0.01, rs_min_dist_range.min, rs_min_dist_range.max))
                    set_option_safe(rs_depth_sensor, rs.option.min_distance, rs_current_min_dist, "min_distance")
                    print(f"[DEPTH] min_distance = {rs_current_min_dist:.3f} m")
                
                # Max distance up/down (i/k)
                if key == ord('i') and rs_max_dist_range:
                    rs_current_max_dist = float(np.clip(rs_current_max_dist + 0.01, rs_max_dist_range.min, rs_max_dist_range.max))
                    set_option_safe(rs_depth_sensor, rs.option.max_distance, rs_current_max_dist, "max_distance")
                    print(f"[DEPTH] max_distance = {rs_current_max_dist:.3f} m")
                if key == ord('k') and rs_max_dist_range:
                    rs_current_max_dist = float(np.clip(rs_current_max_dist - 0.01, rs_max_dist_range.min, rs_max_dist_range.max))
                    set_option_safe(rs_depth_sensor, rs.option.max_distance, rs_current_max_dist, "max_distance")
                    print(f"[DEPTH] max_distance = {rs_current_max_dist:.3f} m")

            # Q = quit without toggling anything further
            if key == ord('q'):
                if recording:
                    # If 0 frames, just restore to initial state without saving
                    if frame_idx == 0:
                        print(f"\n[INFO] Recording stopped at 0 frames (quit). Restoring to initial state.")
                        if temp_seq_root is not None and temp_seq_root.exists():
                            import shutil
                            shutil.rmtree(temp_seq_root)
                            print(f"[INFO] Temporary directory deleted.")
                        # Reset to initial state
                        recording = False
                        frame_idx = 0
                        seq_root = None
                        temp_seq_root = None
                        rs_root = None
                        zed_root = None
                        rs_render_root = None
                        zed_render_root = None
                        waiting_for_save_decision = False
                        break
                    else:
                        # Stop recording and wait for save decision
                        print(f"\n[INFO] Recording stopped at {frame_idx} frames (quit).")
                        print(f"[INFO] Press Y to save, N to discard (in the preview window)")
                        waiting_for_save_decision = True
                        recording = False
                        # Don't break yet, wait for Y/N decision
                elif waiting_for_save_decision:
                    # If already waiting for decision, Q means discard and quit
                    # If 0 frames, just restore to initial state
                    if frame_idx == 0:
                        print(f"[INFO] Recording had 0 frames. Restoring to initial state and quitting.")
                        if temp_seq_root is not None and temp_seq_root.exists():
                            import shutil
                            shutil.rmtree(temp_seq_root)
                            print(f"[INFO] Temporary directory deleted.")
                    else:
                        print(f"[INFO] Discarding recording and quitting...")
                        if temp_seq_root is not None and temp_seq_root.exists():
                            import shutil
                            shutil.rmtree(temp_seq_root)
                            print(f"[INFO] Temporary directory deleted.")
                    break
                else:
                    # Not recording, just quit
                    break

    finally:
        # Clean up: if still recording or waiting for decision, discard (can't ask interactively in finally)
        # Skip cleanup if 0 frames (restore to initial state)
        if (recording or waiting_for_save_decision) and frame_idx > 0 and temp_seq_root is not None and temp_seq_root.exists():
            print(f"\n[INFO] Program exiting. Discarding unsaved recording at {frame_idx} frames.")
            import shutil
            shutil.rmtree(temp_seq_root)
            print(f"[INFO] Temporary directory deleted.")
        elif (recording or waiting_for_save_decision) and frame_idx == 0 and temp_seq_root is not None and temp_seq_root.exists():
            # 0 frames: just clean up without message
            import shutil
            shutil.rmtree(temp_seq_root)
        
        rs_pipeline.stop()
        zed.close()
        cv2.destroyAllWindows()
        print("[INFO] Cameras closed. Done.")


if __name__ == "__main__":
    main()
