import math
import numpy as np
import cv2
import pyzed.sl as sl
from pupil_apriltags import Detector

# ---------------- CONFIG ----------------

FAMILY = "tag36h11"
TAG_SIZE_MM = 60
TAG_SIZE_M = TAG_SIZE_MM / 1000.0

CENTER_TAG_ID = 0  # AprilTag ID at the center of the table (for X, Y position)
USE_MULTI_TAG_DEPTH = True  # Use multiple tags to average depth (Z) for better accuracy
DEPTH_TAG_IDS = [0, 1, 2, 3, 4, 5, 6]  # AprilTag IDs to use for depth averaging (all on same surface)

SAMPLES_TO_AVG = 30  # how many frames to average

# ZED2i stereo baseline (distance between left and right cameras)
ZED2I_BASELINE_MM = 120  # mm

# Coordinate assumptions:
# - WORLD origin is at the center of the AprilTag (tag is now at table center)
# - AprilTag coordinate frame (standard AprilTag orientation):
#     +X axis: to the right when looking at the tag
#     +Y axis: downward in the tag plane
#     +Z axis: pointing OUT from the tag surface (towards the camera)
#
# Note: We are using the LEFT stereo camera of the ZED2i.
#       The right camera is ~120mm to the right of the left camera.
#       All positions reported are for the LEFT camera optical center.


# ---------------- UTILS ----------------

def make_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rotation_matrix_to_euler_xyz(R):
    """Convert rotation matrix to XYZ Euler angles in degrees."""
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    x = math.asin(sy)

    if abs(sy) < 0.9999:
        y = math.atan2(R[2, 1], R[2, 2])
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        y = 0.0
        z = math.atan2(-R[0, 1], R[1, 1])

    return tuple(math.degrees(v) for v in (x, y, z))


def quaternion_to_euler_xyz(qx, qy, qz, qw):
    """
    Convert quaternion to XYZ Euler angles in degrees.
    
    Args:
        qx, qy, qz, qw: Quaternion components
    
    Returns:
        tuple: (pitch, yaw, roll) in degrees
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


# ---------------- MAIN ----------------

def main():
    # ---- Open ZED ----
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080  # 1920 x 1080
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    init.sensors_required = True  # Enable IMU sensors

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED:", status)
        return

    info = zed.get_camera_information()
    calib = info.camera_configuration.calibration_parameters.left_cam
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy
    
    # Get IMU sensor information
    sensors_data = sl.SensorsData()
    imu_available = False
    if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
        imu_available = True

    print("=== ZED LEFT camera info ===")
    print("Resolution:",
          info.camera_configuration.resolution.width,
          "x",
          info.camera_configuration.resolution.height)
    print(f"Intrinsics: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    print(f"ZED2i stereo baseline: {ZED2I_BASELINE_MM} mm")
    
    # Print IMU info if available
    if imu_available:
        imu_data = sensors_data.get_imu_data()
        quat = imu_data.get_pose().get_orientation().get()
        qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
        imu_roll, imu_pitch, imu_yaw = quaternion_to_euler_xyz(qx, qy, qz, qw)
        
        print(f"\n=== IMU Sensor Data (at start) ===")
        print(f"Orientation (Euler angles):")
        print(f"   Roll  (X-axis): {imu_roll:+7.2f}°")
        print(f"   Pitch (Y-axis): {imu_pitch:+7.2f}°")
        print(f"   Yaw   (Z-axis): {imu_yaw:+7.2f}°")
        print(f"Orientation (quaternion): [x={qx:.3f}, y={qy:.3f}, z={qz:.3f}, w={qw:.3f}]")
        print(f"Angular Velocity (deg/s): [{imu_data.get_angular_velocity()[0]:.3f}, "
              f"{imu_data.get_angular_velocity()[1]:.3f}, "
              f"{imu_data.get_angular_velocity()[2]:.3f}]")
        print(f"Linear Acceleration (m/s²): [{imu_data.get_linear_acceleration()[0]:.3f}, "
              f"{imu_data.get_linear_acceleration()[1]:.3f}, "
              f"{imu_data.get_linear_acceleration()[2]:.3f}]")
    else:
        print("\n⚠️ IMU sensor data not available")
    
    print(f"\n📍 Using AprilTag ID {CENTER_TAG_ID} at center of table for X, Y position.")
    if USE_MULTI_TAG_DEPTH:
        print(f"📏 Using multiple tags {DEPTH_TAG_IDS} to average depth (Z) for better accuracy.")
    else:
        print(f"📏 Using single tag for depth measurement.")
    print(f"Collecting {SAMPLES_TO_AVG} frames for averaging. Hold camera steady.\n")

    detector = Detector(
        families=FAMILY,
        nthreads=4,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=False,
    )

    image_mat = sl.Mat()
    Rs = []
    ts = []
    depth_samples = []  # Store depth measurements from multiple tags

    try:
        while len(ts) < SAMPLES_TO_AVG:
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            frame_bgra = image_mat.get_data()
            frame_bgr = frame_bgra[:, :, :3]

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            results = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=TAG_SIZE_M,
            )

            # Find center tag for X, Y position
            center_det = None
            depth_dets = {}  # Dictionary to store depth tags by ID
            
            for r in results:
                if r.tag_id == CENTER_TAG_ID:
                    center_det = r
                if USE_MULTI_TAG_DEPTH and r.tag_id in DEPTH_TAG_IDS:
                    depth_dets[r.tag_id] = r

            if center_det is None:
                print(f"Center tag (ID {CENTER_TAG_ID}) not visible in this frame...")
                continue

            # Get X, Y, Rotation from center tag
            R_cam_tag = center_det.pose_R
            t_cam_tag = center_det.pose_t.reshape(3)

            # We want CAMERA in TAG frame (inverse): T_tag_cam
            R_tag_cam = R_cam_tag.T
            t_tag_cam = -R_cam_tag.T @ t_cam_tag

            # Collect depth (Z) measurements from all visible depth tags
            if USE_MULTI_TAG_DEPTH and len(depth_dets) > 0:
                frame_depths = []
                frame_depth_dict = {}  # Store depth by tag ID
                
                for tag_id, det in depth_dets.items():
                    R_cam_tag_depth = det.pose_R
                    t_cam_tag_depth = det.pose_t.reshape(3)
                    R_tag_cam_depth = R_cam_tag_depth.T
                    t_tag_cam_depth = -R_cam_tag_depth.T @ t_cam_tag_depth
                    depth_z = t_tag_cam_depth[2]  # Z coordinate
                    frame_depths.append(depth_z)
                    frame_depth_dict[tag_id] = depth_z
                
                avg_depth = np.mean(frame_depths)
                depth_samples.append(frame_depth_dict)  # Store dict instead of list
                
                # Replace Z with averaged depth from multiple tags
                t_tag_cam_multi = t_tag_cam.copy()
                t_tag_cam_multi[2] = avg_depth
                
                Rs.append(R_tag_cam)
                ts.append(t_tag_cam_multi)
                
                # Print individual tag depths
                depth_str = ", ".join([f"ID{tid}:{d*1000:.1f}mm" for tid, d in sorted(frame_depth_dict.items())])
                print(f"Sample {len(ts)}/{SAMPLES_TO_AVG}: "
                      f"XY from tag {CENTER_TAG_ID} | Depths: [{depth_str}] "
                      f"→ avg={avg_depth*1000:.1f}mm (std={np.std(frame_depths)*1000:.1f}mm)")
            else:
                # Fall back to single tag
                Rs.append(R_tag_cam)
                ts.append(t_tag_cam)
                
                print(f"Sample {len(ts)}/{SAMPLES_TO_AVG}: "
                      f"cam in TAG frame t={t_tag_cam}")

        # Average translation; take rotation from first (simple but OK)
        t_tag_cam_avg = np.mean(np.stack(ts, axis=0), axis=0)
        R_tag_cam_avg = Rs[0]

        # Conversion factors
        MM_TO_INCH = 0.0393701
        M_TO_INCH = 39.3701

        print("\n" + "="*70)
        print("CAMERA CALIBRATION RESULTS (LEFT Camera Optical Center)")
        print("="*70)
        print("\n🎯 POSITIONING WORKFLOW:")
        print("   Step 1: Position camera center at X, Y, Z coordinates shown below")
        print("   Step 2: THEN apply rotation angles (Pitch, Yaw, Roll)")
        print("="*70)
        
        # Show depth measurement statistics if using multi-tag
        if USE_MULTI_TAG_DEPTH and len(depth_samples) > 0:
            # Organize depths by tag ID
            depths_by_tag = {}
            all_depths = []
            
            for frame_depth_dict in depth_samples:
                for tag_id, depth in frame_depth_dict.items():
                    if tag_id not in depths_by_tag:
                        depths_by_tag[tag_id] = []
                    depths_by_tag[tag_id].append(depth)
                    all_depths.append(depth)
            
            all_depths = np.array(all_depths)
            
            print(f"\n📊 Depth Measurement Statistics:")
            print(f"   Overall: {len(all_depths)} total measurements")
            print(f"   Mean depth: {np.mean(all_depths)*1000:.2f} mm ({np.mean(all_depths)*M_TO_INCH:.2f} in)")
            print(f"   Std deviation: {np.std(all_depths)*1000:.2f} mm")
            print(f"   Range: {(np.max(all_depths) - np.min(all_depths))*1000:.2f} mm")
            
            print(f"\n📍 Individual Tag Depth Statistics:")
            for tag_id in sorted(depths_by_tag.keys()):
                tag_depths = np.array(depths_by_tag[tag_id])
                print(f"   Tag ID {tag_id}: {len(tag_depths)} samples")
                print(f"      Mean: {np.mean(tag_depths)*1000:.2f} mm ({np.mean(tag_depths)*M_TO_INCH:.2f} in)")
                print(f"      Std:  {np.std(tag_depths)*1000:.2f} mm")
                print(f"      Min:  {np.min(tag_depths)*1000:.2f} mm")
                print(f"      Max:  {np.max(tag_depths)*1000:.2f} mm")
                print(f"      Range: {(np.max(tag_depths) - np.min(tag_depths))*1000:.2f} mm")
            
            print(f"\n   ✓ Using averaged depth from {len(depths_by_tag)} tags improves accuracy!")
        
        # In AprilTag frame (standard orientation):
        # X = right, Y = down, Z = out from tag (toward camera)
        x_m, y_m, z_m = t_tag_cam_avg
        
        print(f"\n📍 Position in AprilTag coordinate frame:")
        if USE_MULTI_TAG_DEPTH:
            print(f"   (X, Y from tag {CENTER_TAG_ID} | Z averaged from tags {DEPTH_TAG_IDS})")
        print(f"   X = {x_m:+.4f} m = {x_m*1000:+7.1f} mm = {x_m*M_TO_INCH:+6.2f} in  (+ is RIGHT of tag)")
        print(f"   Y = {y_m:+.4f} m = {y_m*1000:+7.1f} mm = {y_m*M_TO_INCH:+6.2f} in  (+ is DOWN from tag center)")
        print(f"   Z = {z_m:+.4f} m = {z_m*1000:+7.1f} mm = {z_m*M_TO_INCH:+6.2f} in  (+ is AWAY from tag surface)")
        
        total_dist_m = np.linalg.norm(t_tag_cam_avg)
        print(f"\n📏 Distance from tag center: {total_dist_m:.4f} m = {total_dist_m*1000:.1f} mm = {total_dist_m*M_TO_INCH:.2f} in")
        
        # Compute Euler angles for orientation
        euler_x, euler_y, euler_z = rotation_matrix_to_euler_xyz(R_tag_cam_avg)
        
        print(f"\n🔄 Camera Orientation (from AprilTag detection):")
        print(f"   Pitch (X-axis rotation): {euler_x:+7.2f}°")
        print(f"   Yaw   (Y-axis rotation): {euler_y:+7.2f}°")
        print(f"   Roll  (Z-axis rotation): {euler_z:+7.2f}°")
        
        # Show IMU orientation comparison if available
        if imu_available:
            # Get current IMU data
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                imu_data = sensors_data.get_imu_data()
                quat = imu_data.get_pose().get_orientation().get()
                qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
                imu_roll, imu_pitch, imu_yaw = quaternion_to_euler_xyz(qx, qy, qz, qw)
                
                print(f"\n🔄 IMU Orientation (for comparison):")
                print(f"   Roll  (X-axis): {imu_roll:+7.2f}°")
                print(f"   Pitch (Y-axis): {imu_pitch:+7.2f}°")
                print(f"   Yaw   (Z-axis): {imu_yaw:+7.2f}°")
                
                print(f"\n📊 Orientation Difference (AprilTag vs IMU):")
                print(f"   Pitch difference: {abs(euler_x - imu_pitch):6.2f}°")
                print(f"   Yaw difference:   {abs(euler_y - imu_yaw):6.2f}°")
                print(f"   Roll difference:  {abs(euler_z - imu_roll):6.2f}°")
        
        print("\n" + "="*70)
        print("SETUP INSTRUCTIONS & VERIFICATION")
        print("="*70)
        print("\n📐 TO SET UP CAMERA AT THIS POSITION:")
        print("   STEP 1 - SET POSITION (X, Y, Z):")
        print(f"      • Move LEFT camera lens {abs(x_m*1000):.1f}mm ({abs(x_m*M_TO_INCH):.2f}in) to the {'RIGHT' if x_m > 0 else 'LEFT'} from tag {CENTER_TAG_ID} center")
        print(f"      • Move it {abs(y_m*1000):.1f}mm ({abs(y_m*M_TO_INCH):.2f}in) {'BELOW' if y_m > 0 else 'ABOVE'} tag center")
        print(f"      • Move it {z_m*1000:.1f}mm ({z_m*M_TO_INCH:.2f}in) away from tag surface (perpendicular)")
        if USE_MULTI_TAG_DEPTH and len(depth_samples) > 0:
            print(f"         (Z is averaged from {len(depth_samples)} frames using tags {DEPTH_TAG_IDS})")
        print(f"\n   STEP 2 - APPLY ROTATION (after position is set):")
        print(f"      • Pitch (rotate around X-axis): {euler_x:+.2f}°")
        print(f"      • Yaw   (rotate around Y-axis): {euler_y:+.2f}°")
        print(f"      • Roll  (rotate around Z-axis): {euler_z:+.2f}°")
        print(f"\n📝 PHYSICAL VERIFICATION:")
        print(f"   1. Horizontal from tag {CENTER_TAG_ID} to LEFT lens: {abs(x_m*1000):.1f}mm ({abs(x_m*M_TO_INCH):.2f}in) {'RIGHT' if x_m > 0 else 'LEFT'}")
        print(f"   2. Vertical from tag {CENTER_TAG_ID} to lens: {abs(y_m*1000):.1f}mm ({abs(y_m*M_TO_INCH):.2f}in) {'BELOW' if y_m > 0 else 'ABOVE'}")
        print(f"   3. Perpendicular from tag surface to lens: {z_m*1000:.1f}mm ({z_m*M_TO_INCH:.2f}in)")
        print(f"   4. Note: RIGHT camera is ~120mm (~4.72in) to the right of LEFT camera")
        print("\n💡 CONFIG: To change which tags to use, edit:")
        print(f"   CENTER_TAG_ID = {CENTER_TAG_ID}  (for X, Y position)")
        print(f"   DEPTH_TAG_IDS = {DEPTH_TAG_IDS}  (for Z averaging)")
        print(f"   USE_MULTI_TAG_DEPTH = {USE_MULTI_TAG_DEPTH}")
        print("="*70)

    finally:
        zed.close()
        print("\nZED closed.")

if __name__ == "__main__":
    main()
