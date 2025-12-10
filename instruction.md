APRILTAG-BASED MULTICAMERA CALIBRATION + HAMER + FOUNDATIONPOSE PIPELINE
============================================================================

GOAL:
- Use one AprilTag placed on the table as the world origin.
- Calibrate extrinsics for ZED2i and RealSense cameras.
- Use HaMeR for 2D keypoints from both cameras.
- Fuse them using triangulation (RANSAC) to get 3D hand trajectory.
- Use FoundationPose for object pose in the same world frame.
- Achieve hand-object contact trajectory for imitation learning.

----------------------------------------------------------------------------

APRILTAG SPECIFICATIONS (PRINTING)
----------------------------------
Use this exact parameter set from the website:

- Tag Family: tag36h11
- Tag ID: 0
- Tag Size (black square): 160 mm
- Total Size (including white border): 200 mm

These values fit on A4 paper and give maximum stable calibration.

IMPORTANT:
- When performing solvePnP or pose estimation, use tag_size = 0.16 meters.
- The white border is NOT included in the tag size used in pose estimation.

Print at 100 percent scaling, do NOT "fit to page".

----------------------------------------------------------------------------

WHAT COORDINATE SYSTEM WE USE
-----------------------------
- The AprilTag defines the WORLD coordinate frame.
- The cameras give you CAMERA coordinate frames.
- You compute T_world_from_camera for each camera via AprilTag detection.
- Once you have these, both HaMeR and FoundationPose outputs can be transformed into the world frame.

This ensures that:
hand trajectory, object trajectory, and your robot motion all share the same coordinate system.

----------------------------------------------------------------------------

EXTRINSICS CALIBRATION PIPELINE
-------------------------------
1. Print the AprilTag (as described above).
2. Tape it flat on the table at the desired world origin.
3. Capture at least 20 RGB images from each camera with the tag visible.
4. Detect AprilTag corners using apriltag or pupil-apriltags.
5. Use factory intrinsics for ZED2i and RealSense.
6. SolvePnP for each camera:
   - Inputs: 2D AprilTag corners, 3D AprilTag corners (known from tag size), intrinsics.
   - Output: Rotation and translation of the camera relative to the AprilTag.
7. Save each camera extrinsic as 4x4 matrix:
   camX_extrinsics.npy

This file must contain T_world_from_camera.

Store them under:
calibration/cam1_extrinsics.npy
calibration/cam2_extrinsics.npy

Likewise save intrinsics as:
calibration/cam1_intrinsics.npy
calibration/cam2_intrinsics.npy

Once this is done, calibration NEVER needs to be redone unless you move the cameras.

----------------------------------------------------------------------------

HAMER 2D KEYPOINT EXTRACTION PIPELINE
-------------------------------------
For each camera:
- Run HaMeR on the RGB image sequence.
- Save:
  frame_00000_righthand_keypoints.npy
  frame_00000_righthand_mano_params.npy
  ...
into:
traj_folder/cam1/processed/
traj_folder/cam2/processed/

These contain 2D keypoints in pixel coordinates.

----------------------------------------------------------------------------

MULTI-CAMERA TRIANGULATION USING RANSAC
----------------------------------------
You will use the triangulate_with_ransac function from DemoDiffusion with modifications:

Input:
- keypoints_list: list of 2x (21x2) arrays for the two cameras.
- intrinsics: [cam1_intrinsic_matrix, cam2_intrinsic_matrix]
- transforms: [cam1_extrinsic, cam2_extrinsic]

Process:
- Build projection matrices P = K * [R|t].
- For each keypoint index i:
    * Gather available 2D detections.
    * Randomly sample 2 cameras.
    * Solve linear triangulation via SVD.
    * Compute reprojection error.
    * Count inliers.
    * Keep best hypothesis.
- After RANSAC, refine using all inliers.

Output:
- kp3d: (21, 3) 3D world coordinates per frame.

Save to:
traj_folder/processed_3d/righthand_3d_keypoints.npy

----------------------------------------------------------------------------

FOUNDATIONPOSE OBJECT POSE
--------------------------
Run FoundationPose on one camera (ZED or RealSense).
Obtain object pose in that camera's frame: T_obj_from_cam.

Convert to world frame:
T_world_from_obj = T_world_from_cam * T_obj_from_cam

This gives you the object's full 6D trajectory aligned to the same world frame as HaMeR.

----------------------------------------------------------------------------

FULL TRAJECTORY ALIGNMENT
--------------------------
After triangulation:
- HaMeR hand keypoints are in the world frame.
- FoundationPose object pose is in the world frame.
- You now have consistent 3D trajectories for both.

Robot imitation learning requires:
robot_EE_target[t] = hand_position_world
object_state[t]   = object_pose_world

----------------------------------------------------------------------------

RECOMMENDED TAG SIZE LOGIC (IMPORTANT)
--------------------------------------
Use Tag Size = 160 mm because:
- Fits safely on A4.
- Larger tag means higher pose accuracy.
- ZED2i wide lens + RealSense noise benefits from larger corner spread.

In calibration code use:
tag_size = 0.16  # meters

----------------------------------------------------------------------------

IMPORTANT FILES YOU MUST GENERATE
----------------------------------
calibration/
    cam1_intrinsics.npy
    cam2_intrinsics.npy
    cam1_extrinsics.npy   (4x4: T_world_from_cam1)
    cam2_extrinsics.npy   (4x4: T_world_from_cam2)

traj_xxx/
    cam1/<images>
    cam2/<images>
    cam1/processed/*_righthand_keypoints.npy
    cam2/processed/*_righthand_keypoints.npy

traj_xxx/processed_3d/
    righthand_3d_keypoints.npy
    eef_pose.npy              (optional)
    retarget_gripper_action.npy (optional)

----------------------------------------------------------------------------

EXPECTED TIME TO COMPLETE
-------------------------
With Cursor + your existing FP pipeline:
- Print tag: 5 minutes
- Capture calibration images: 20 minutes
- Write AprilTag solvePnP script: 1 hour
- Write triangulation code integration: 2 hours
- Test full pipeline: 1-2 hours

Total realistic time: 4-5 hours.

This is completely achievable in one day with focus.

============================================================================
END OF README
