#!/usr/bin/env python3
"""
AprilTag-based camera extrinsics calibration.

This script detects an AprilTag (tag36h11, ID 0) placed at the world origin
and computes the camera extrinsics (T_world_from_camera) for both ZED2i and
RealSense D435 cameras.

WORLD COORDINATE FRAME (with USE_ROBOTICS_FRAME = True):
- Origin: Center of AprilTag on table
- X-axis (RED): Points in the direction the tag's X arrow points
- Y-axis (GREEN): Points LEFT (90° counterclockwise from X when viewed from above)
- Z-axis (BLUE): Points UP (opposite gravity)

HOW TO ORIENT THE APRILTAG:
1. Place tag flat on table, face-up (so cameras above can see it)
2. Rotate the tag so its X-axis (red arrow) points in your desired "forward" direction
3. The world frame will then have:
   - X = forward (where the tag's X points)
   - Y = left
   - Z = up

Output files:
    cam1_extrinsics.npy - ZED2i 4x4 transformation matrix (T_world_from_cam1)
    cam2_extrinsics.npy - RealSense 4x4 transformation matrix (T_world_from_cam2)

Usage:
    python april_extrinsics.py

Controls:
    SPACE - Capture current frame and compute extrinsics for both cameras
    S     - Save extrinsics to .npy files
    Q     - Quit without saving
    ESC   - Quit without saving

Requirements:
    - pupil-apriltags (pip install pupil-apriltags)
    - AprilTag printed: tag36h11, ID 0, 160mm black square (200mm total with border)
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple

import pyrealsense2 as rs
import pyzed.sl as sl

# Use pupil-apriltags for detection
try:
    from pupil_apriltags import Detector
except ImportError:
    print("[ERROR] pupil-apriltags not installed. Run: pip install pupil-apriltags")
    exit(1)


# Camera resolution settings (must match record.py)
RS_WIDTH = 848
RS_HEIGHT = 480
RS_FPS = 30

Z_WIDTH = 1280
Z_HEIGHT = 720
Z_FPS = 30

# AprilTag settings (from instruction.md)
TAG_FAMILY = "tag36h11"
TAG_ID = 0
TAG_SIZE_M = 0.16  # Black square size in meters (160mm)

# =============================================================================
# WORLD FRAME CONVENTION
# =============================================================================
# The AprilTag library uses: Z pointing OUT of tag (toward camera)
# For robotics, we typically want: Z pointing UP (opposite gravity)
#
# Physical setup assumption:
#   - AprilTag is flat on the table, face-up
#   - Cameras are ABOVE the table looking DOWN at the tag
#
# To convert from AprilTag frame to standard robotics world frame:
#   - Rotate 180° around X axis to flip Z from "toward camera" to "up"
#
# Set USE_ROBOTICS_FRAME = True to apply this correction
# =============================================================================
USE_ROBOTICS_FRAME = True  # Set to True for Z-up world frame

def get_world_frame_correction() -> np.ndarray:
    """
    Get the transformation to convert AprilTag frame to desired world frame.
    
    AprilTag convention:
        - Z points OUT of tag (toward camera looking at it)
        - X points right (when looking at tag)
        - Y points down (when looking at tag)
    
    When tag is on table and camera looks down:
        - Tag Z points toward camera (up from table)
        - But detected pose has camera at negative Z
    
    For robotics world frame (Z-up):
        - We want Z pointing UP (opposite gravity)
        - X pointing FORWARD (along table)
        - Y pointing LEFT
    
    This requires rotating 180° around the tag's X axis.
    
    Returns:
        4x4 transformation matrix
    """
    if not USE_ROBOTICS_FRAME:
        return np.eye(4)
    
    # Rotation 180° around X axis: flips Y and Z
    # This makes:
    #   - Z point UP (was pointing toward camera/down in world)
    #   - Y point backward (was forward)
    # 
    # After this, rotate the physical tag so X points where you want "forward"
    R_flip = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float64)
    
    T_correction = np.eye(4)
    T_correction[:3, :3] = R_flip
    
    return T_correction


class CameraCalibrator:
    """
    Handles AprilTag detection and extrinsics computation for both cameras.
    """
    
    def __init__(self):
        """Initialize cameras and AprilTag detector."""
        self.output_dir = Path(__file__).parent
        
        # Initialize AprilTag detector
        self.detector = Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25,
        )
        
        # Camera intrinsics (will be loaded or computed)
        self.zed_K: Optional[np.ndarray] = None
        self.rs_K: Optional[np.ndarray] = None
        
        # Extrinsics results
        self.zed_extrinsics: Optional[np.ndarray] = None
        self.rs_extrinsics: Optional[np.ndarray] = None
        
        # Camera objects
        self.zed: Optional[sl.Camera] = None
        self.rs_pipeline: Optional[rs.pipeline] = None
        
        self._setup_cameras()
    
    def _setup_cameras(self):
        """Initialize both cameras and get intrinsics."""
        print("=" * 60)
        print("APRILTAG EXTRINSICS CALIBRATION")
        print("=" * 60)
        print()
        
        # Setup ZED2i
        print("[INFO] Setting up ZED2i camera...")
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = Z_FPS
        init_params.depth_mode = sl.DEPTH_MODE.NONE  # Don't need depth for calibration
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED: {err}")
        
        # Get ZED intrinsics
        info = self.zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters.left_cam
        self.zed_fx, self.zed_fy = calib.fx, calib.fy
        self.zed_cx, self.zed_cy = calib.cx, calib.cy
        self.zed_K = np.array([
            [self.zed_fx, 0.0, self.zed_cx],
            [0.0, self.zed_fy, self.zed_cy],
            [0.0, 0.0, 1.0]
        ])
        print(f"[INFO] ZED intrinsics: fx={self.zed_fx:.2f}, fy={self.zed_fy:.2f}, "
              f"cx={self.zed_cx:.2f}, cy={self.zed_cy:.2f}")
        
        # Setup RealSense
        print("[INFO] Setting up RealSense D435...")
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.rgb8, RS_FPS)
        
        profile = self.rs_pipeline.start(config)
        
        # Get RealSense intrinsics
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        self.rs_fx_orig = intrinsics.fx
        self.rs_fy_orig = intrinsics.fy
        self.rs_cx_orig = intrinsics.ppx
        self.rs_cy_orig = intrinsics.ppy
        
        # RealSense is mounted upside down - adjust principal point for 180° rotation
        self.rs_fx = self.rs_fx_orig
        self.rs_fy = self.rs_fy_orig
        self.rs_cx = RS_WIDTH - self.rs_cx_orig
        self.rs_cy = RS_HEIGHT - self.rs_cy_orig
        
        self.rs_K = np.array([
            [self.rs_fx, 0.0, self.rs_cx],
            [0.0, self.rs_fy, self.rs_cy],
            [0.0, 0.0, 1.0]
        ])
        print(f"[INFO] RealSense intrinsics (180° rotated): fx={self.rs_fx:.2f}, fy={self.rs_fy:.2f}, "
              f"cx={self.rs_cx:.2f}, cy={self.rs_cy:.2f}")
        
        # Pre-allocate ZED image buffer
        self.zed_image = sl.Mat()
        
        print()
        print("[INFO] Both cameras initialized successfully.")
        print()
        print("CONTROLS:")
        print("  SPACE - Capture and compute extrinsics")
        print("  S     - Save extrinsics to .npy files")
        print("  Q/ESC - Quit")
        print()
    
    def _get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture frames from both cameras.
        
        Returns:
            Tuple of (zed_bgr, rs_bgr) images, or None if capture failed.
        """
        zed_bgr = None
        rs_bgr = None
        
        # Capture ZED frame
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.zed_image, sl.VIEW.LEFT)
            zed_bgra = self.zed_image.get_data()
            zed_bgr = cv2.cvtColor(zed_bgra, cv2.COLOR_BGRA2BGR)
        
        # Capture RealSense frame
        frames = self.rs_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            color = np.asanyarray(color_frame.get_data())
            rs_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            # Rotate 180 degrees (camera is mounted upside down)
            rs_bgr = cv2.rotate(rs_bgr, cv2.ROTATE_180)
        
        return zed_bgr, rs_bgr
    
    def _detect_apriltag(self, gray: np.ndarray, K: np.ndarray, 
                          camera_name: str) -> Optional[np.ndarray]:
        """
        Detect AprilTag and compute T_world_from_camera.
        
        Args:
            gray: Grayscale image
            K: 3x3 camera intrinsic matrix
            camera_name: Name for logging
            
        Returns:
            4x4 transformation matrix T_world_from_camera, or None if not detected.
        """
        # Detect AprilTags
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[fx, fy, cx, cy],
            tag_size=TAG_SIZE_M
        )
        
        # Find tag with ID 0
        target_detection = None
        for det in detections:
            if det.tag_id == TAG_ID:
                target_detection = det
                break
        
        if target_detection is None:
            print(f"[{camera_name}] AprilTag ID {TAG_ID} not detected")
            return None
        
        # Get pose from detection (T_camera_from_tag)
        R_cam_tag = target_detection.pose_R  # 3x3 rotation
        t_cam_tag = target_detection.pose_t.flatten()  # 3x1 translation
        
        # Build 4x4 T_camera_from_tag
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = R_cam_tag
        T_cam_tag[:3, 3] = t_cam_tag
        
        # T_tag_from_camera = inverse(T_camera_from_tag)
        T_tag_cam = np.linalg.inv(T_cam_tag)
        
        # Apply world frame correction (e.g., rotate to make Z point up)
        T_correction = get_world_frame_correction()
        T_world_cam = T_correction @ T_tag_cam
        
        # Print detection info
        print(f"[{camera_name}] AprilTag detected!")
        print(f"  Tag distance: {np.linalg.norm(t_cam_tag):.3f} m")
        print(f"  Camera position in world: {T_world_cam[:3, 3]}")
        if USE_ROBOTICS_FRAME:
            print(f"  (Using robotics frame: Z-up)")
        
        return T_world_cam
    
    def _draw_detection(self, img: np.ndarray, gray: np.ndarray, 
                         K: np.ndarray, camera_name: str) -> np.ndarray:
        """
        Draw AprilTag detection on image.
        
        Args:
            img: BGR image to draw on
            gray: Grayscale image for detection
            K: Camera intrinsic matrix
            camera_name: Camera name for display
            
        Returns:
            Image with detection overlay
        """
        vis = img.copy()
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[fx, fy, cx, cy],
            tag_size=TAG_SIZE_M
        )
        
        target_found = False
        
        for det in detections:
            # Draw corners
            corners = det.corners.astype(int)
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                color = (0, 255, 0) if det.tag_id == TAG_ID else (0, 0, 255)
                cv2.line(vis, pt1, pt2, color, 2)
            
            # Draw center
            center = tuple(det.center.astype(int))
            cv2.circle(vis, center, 5, (0, 255, 255), -1)
            
            # Draw tag ID
            cv2.putText(vis, f"ID: {det.tag_id}", 
                       (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw axis if this is our target tag
            if det.tag_id == TAG_ID:
                target_found = True
                R = det.pose_R
                t = det.pose_t.flatten()
                
                # Build T_camera_from_tag
                T_cam_tag = np.eye(4)
                T_cam_tag[:3, :3] = R
                T_cam_tag[:3, 3] = t
                
                # Get world frame correction and compute T_camera_from_world
                T_correction = get_world_frame_correction()
                # T_cam_world = T_cam_tag @ inv(T_correction)
                T_cam_world = T_cam_tag @ np.linalg.inv(T_correction)
                
                # Project WORLD axis endpoints (after correction)
                axis_length = TAG_SIZE_M * 0.5
                axis_pts_world = np.array([
                    [0, 0, 0, 1],
                    [axis_length, 0, 0, 1],  # X (red) - FORWARD
                    [0, axis_length, 0, 1],  # Y (green) - LEFT
                    [0, 0, axis_length, 1],  # Z (blue) - UP
                ]).T
                
                # Transform to camera frame
                axis_cam = T_cam_world @ axis_pts_world
                axis_cam = axis_cam[:3, :]  # Remove homogeneous coord
                
                # Project to image
                axis_2d = K @ axis_cam
                axis_2d = axis_2d[:2] / axis_2d[2:3]
                axis_2d = axis_2d.T.astype(int)
                
                origin = tuple(axis_2d[0])
                cv2.arrowedLine(vis, origin, tuple(axis_2d[1]), (0, 0, 255), 3)  # X red (forward)
                cv2.arrowedLine(vis, origin, tuple(axis_2d[2]), (0, 255, 0), 3)  # Y green (left)
                cv2.arrowedLine(vis, origin, tuple(axis_2d[3]), (255, 0, 0), 3)  # Z blue (up)
                
                # Add axis labels
                cv2.putText(vis, "X", tuple(axis_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(vis, "Y", tuple(axis_2d[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(vis, "Z", tuple(axis_2d[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add camera name
        cv2.putText(vis, camera_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add status
        if target_found:
            status = "TAG DETECTED - Press SPACE to capture"
            color = (0, 255, 0)
        else:
            status = "Looking for tag ID 0..."
            color = (0, 0, 255)
        cv2.putText(vis, status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis
    
    def capture_extrinsics(self, zed_bgr: np.ndarray, rs_bgr: np.ndarray) -> bool:
        """
        Capture and compute extrinsics for both cameras.
        
        Args:
            zed_bgr: ZED BGR image
            rs_bgr: RealSense BGR image (already rotated 180°)
            
        Returns:
            True if both cameras detected the tag successfully.
        """
        print()
        print("=" * 40)
        print("CAPTURING EXTRINSICS...")
        print("=" * 40)
        
        success = True
        
        # Process ZED
        if zed_bgr is not None:
            zed_gray = cv2.cvtColor(zed_bgr, cv2.COLOR_BGR2GRAY)
            T = self._detect_apriltag(zed_gray, self.zed_K, "ZED")
            if T is not None:
                self.zed_extrinsics = T
            else:
                success = False
        else:
            print("[ZED] No frame captured")
            success = False
        
        # Process RealSense
        if rs_bgr is not None:
            rs_gray = cv2.cvtColor(rs_bgr, cv2.COLOR_BGR2GRAY)
            T = self._detect_apriltag(rs_gray, self.rs_K, "RealSense")
            if T is not None:
                self.rs_extrinsics = T
            else:
                success = False
        else:
            print("[RealSense] No frame captured")
            success = False
        
        if success:
            print()
            print("[SUCCESS] Both cameras detected the AprilTag!")
            print("Press 'S' to save extrinsics.")
        else:
            print()
            print("[FAILED] Could not detect tag in all cameras.")
            print("Ensure AprilTag ID 0 is visible to both cameras.")
        
        return success
    
    def save_extrinsics(self) -> bool:
        """
        Save extrinsics to .npy files.
        
        Returns:
            True if saved successfully.
        """
        if self.zed_extrinsics is None or self.rs_extrinsics is None:
            print("[ERROR] Extrinsics not computed yet. Press SPACE to capture first.")
            return False
        
        print()
        print("=" * 40)
        print("SAVING EXTRINSICS...")
        print("=" * 40)
        
        # Save ZED extrinsics (cam1)
        zed_path = self.output_dir / "cam1_extrinsics.npy"
        np.save(zed_path, self.zed_extrinsics)
        print(f"[SAVED] ZED extrinsics -> {zed_path}")
        print(f"        T_world_from_cam1 =")
        print(self.zed_extrinsics)
        print()
        
        # Save RealSense extrinsics (cam2)
        rs_path = self.output_dir / "cam2_extrinsics.npy"
        np.save(rs_path, self.rs_extrinsics)
        print(f"[SAVED] RealSense extrinsics -> {rs_path}")
        print(f"        T_world_from_cam2 =")
        print(self.rs_extrinsics)
        print()
        
        print("[SUCCESS] Extrinsics saved!")
        print("These matrices transform points from camera frame to world frame.")
        
        return True
    
    def run(self):
        """Main calibration loop."""
        cv2.namedWindow("AprilTag Calibration", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # Capture frames
                zed_bgr, rs_bgr = self._get_frames()
                
                # Create visualization
                if zed_bgr is not None and rs_bgr is not None:
                    zed_gray = cv2.cvtColor(zed_bgr, cv2.COLOR_BGR2GRAY)
                    rs_gray = cv2.cvtColor(rs_bgr, cv2.COLOR_BGR2GRAY)
                    
                    zed_vis = self._draw_detection(zed_bgr, zed_gray, self.zed_K, "ZED (cam1)")
                    rs_vis = self._draw_detection(rs_bgr, rs_gray, self.rs_K, "RealSense (cam2)")
                    
                    # Resize to match heights
                    h1, w1 = zed_vis.shape[:2]
                    h2, w2 = rs_vis.shape[:2]
                    
                    if h1 != h2:
                        scale = h1 / h2
                        rs_vis = cv2.resize(rs_vis, (int(w2 * scale), h1))
                    
                    vis = np.hstack([zed_vis, rs_vis])
                    
                    # Add instructions
                    cv2.putText(vis, "SPACE: Capture | S: Save | Q: Quit", 
                               (10, vis.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("AprilTag Calibration", vis)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # SPACE - capture
                    if zed_bgr is not None and rs_bgr is not None:
                        self.capture_extrinsics(zed_bgr, rs_bgr)
                
                elif key == ord('s') or key == ord('S'):  # S - save
                    self.save_extrinsics()
                
                elif key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC - quit
                    print("[INFO] Quitting...")
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up camera resources."""
        print("[INFO] Closing cameras...")
        
        if self.zed is not None:
            self.zed.close()
        
        if self.rs_pipeline is not None:
            self.rs_pipeline.stop()
        
        cv2.destroyAllWindows()
        print("[INFO] Done.")


def main():
    """Main entry point."""
    calibrator = CameraCalibrator()
    calibrator.run()


if __name__ == "__main__":
    main()

