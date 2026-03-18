#!/usr/bin/env python3
"""
3D Visualization of Hand + Object Trajectories.

DEPRECATED: Use postprocess.py instead — it runs HaMeR + triangulation + EEF
and automatically generates the 3D visualization:

    python record/hamer/postprocess.py --data_path data/cup_grasp/001 --mesh cup

This script is kept for reference but expects the old data layout and AprilTag
calibration. The new pipeline uses FP-based calibration and object-centric frame.
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

# Try to import trimesh for mesh loading
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# FoundationPose mesh directory
FP_MESH_DIR = Path("/home/jeff/Desktop/FoundationPose/object")

# Keypoint names and colors for hand trajectory
KP_NAMES = ["Wrist", "Thumb", "Index", "Middle", "Ring", "Pinky"]
KP_COLORS = ["#00FF00", "#FF0000", "#FF8800", "#FFFF00", "#00FFFF", "#FF00FF"]

# Table dimensions: 36" x 36" in meters
TABLE_SIZE_INCHES = 36
TABLE_SIZE_M = TABLE_SIZE_INCHES * 0.0254  # ~0.9144 meters


def draw_table(ax, table_size: float = TABLE_SIZE_M, z_offset: float = 0.0):
    """
    Draw a table surface centered at the origin.
    
    Args:
        ax: matplotlib 3D axis
        table_size: table side length in meters
        z_offset: z position of table surface (default 0)
    """
    half = table_size / 2
    
    # Table corners
    corners = np.array([
        [-half, -half, z_offset],
        [half, -half, z_offset],
        [half, half, z_offset],
        [-half, half, z_offset]
    ])
    
    # Create table surface
    verts = [[corners[0], corners[1], corners[2], corners[3]]]
    table = Poly3DCollection(verts, alpha=0.3, facecolor='brown', edgecolor='saddlebrown', linewidth=2)
    ax.add_collection3d(table)
    
    # Draw grid lines on table (every 6 inches = 0.1524m)
    grid_spacing = 6 * 0.0254  # 6 inches in meters
    for x in np.arange(-half, half + grid_spacing/2, grid_spacing):
        ax.plot([x, x], [-half, half], [z_offset, z_offset], 
                color='saddlebrown', alpha=0.3, linewidth=0.5)
    for y in np.arange(-half, half + grid_spacing/2, grid_spacing):
        ax.plot([-half, half], [y, y], [z_offset, z_offset], 
                color='saddlebrown', alpha=0.3, linewidth=0.5)
    
    # Draw table border
    ax.plot([-half, half, half, -half, -half], 
            [-half, -half, half, half, -half],
            [z_offset, z_offset, z_offset, z_offset, z_offset],
            color='saddlebrown', linewidth=3)
    
    # Draw origin marker
    ax.scatter([0], [0], [z_offset], color='black', s=100, marker='x', linewidths=2, zorder=10)
    
    # Draw coordinate axes
    axis_len = 0.1
    ax.quiver(0, 0, z_offset, axis_len, 0, 0, color='red', arrow_length_ratio=0.15, linewidth=2)
    ax.quiver(0, 0, z_offset, 0, axis_len, 0, color='green', arrow_length_ratio=0.15, linewidth=2)
    ax.quiver(0, 0, z_offset, 0, 0, axis_len, color='blue', arrow_length_ratio=0.15, linewidth=2)


def load_hand_trajectory(data_path: Path) -> np.ndarray:
    """
    Load hand trajectory from HaMeR output.
    
    Args:
        data_path: Path to data folder
        
    Returns:
        np.ndarray: Shape (N, 6, 3) - wrist + 5 fingertips
    """
    traj_path = data_path / "traj" / "hand_trajectory.npy"
    if not traj_path.exists():
        print(f"[WARN] Hand trajectory not found: {traj_path}")
        return None
    
    traj = np.load(traj_path)
    print(f"[INFO] Loaded hand trajectory: {traj.shape}")
    return traj


def load_object_trajectory(data_path: Path, camera: str = "zed") -> np.ndarray:
    """
    Load object trajectory from FoundationPose output.
    
    Args:
        data_path: Path to data folder
        camera: Camera used for FP (zed or realsense)
        
    Returns:
        np.ndarray: Shape (N, 3) - object positions
    """
    traj_path = data_path / "traj" / "FP" / camera / "object_trajectory.npy"
    if not traj_path.exists():
        print(f"[WARN] Object trajectory not found: {traj_path}")
        return None
    
    traj = np.load(traj_path)
    print(f"[INFO] Loaded object trajectory: {traj.shape}")
    return traj


def load_object_poses(data_path: Path, camera: str = "zed") -> np.ndarray:
    """
    Load full object poses (4x4 matrices) from FoundationPose.
    
    Args:
        data_path: Path to data folder
        camera: Camera used for FP
        
    Returns:
        np.ndarray: Shape (N, 4, 4) - pose matrices
    """
    poses_path = data_path / "traj" / "FP" / camera / "object_poses.npy"
    if not poses_path.exists():
        print(f"[WARN] Object poses not found: {poses_path}")
        return None
    
    poses = np.load(poses_path)
    print(f"[INFO] Loaded object poses: {poses.shape}")
    return poses


def load_mesh_points(mesh_name: str, n_points: int = 500) -> np.ndarray:
    """
    Load mesh and sample points from surface.
    
    Args:
        mesh_name: Name of mesh folder in FoundationPose/object/
        n_points: Number of points to sample
        
    Returns:
        np.ndarray: Shape (n_points, 3) - sampled points in object frame
    """
    if not HAS_TRIMESH:
        print("[WARN] trimesh not installed, cannot load mesh")
        return None
    
    mesh_dir = FP_MESH_DIR / mesh_name
    if not mesh_dir.exists():
        print(f"[WARN] Mesh directory not found: {mesh_dir}")
        return None
    
    # Find .obj file
    obj_files = list(mesh_dir.glob("*.obj"))
    if not obj_files:
        print(f"[WARN] No .obj file found in {mesh_dir}")
        return None
    
    mesh = trimesh.load(str(obj_files[0]))
    print(f"[INFO] Loaded mesh: {obj_files[0].name}")
    
    # Sample points from surface
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    print(f"[INFO] Sampled {len(points)} points from mesh surface")
    
    return points


def transform_mesh_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Transform mesh points by a 4x4 pose matrix.
    
    Args:
        points: (N, 3) points in object frame
        pose: 4x4 transformation matrix
        
    Returns:
        np.ndarray: (N, 3) transformed points
    """
    N = len(points)
    ones = np.ones((N, 1))
    points_homo = np.hstack([points, ones])  # (N, 4)
    transformed = (pose @ points_homo.T).T  # (N, 4)
    return transformed[:, :3]


def transform_to_world(positions_cam: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """
    Transform positions from camera frame to world frame.
    
    Args:
        positions_cam: (N, 3) positions in camera frame
        T_world_cam: 4x4 transformation matrix (world from camera)
        
    Returns:
        np.ndarray: (N, 3) positions in world frame
    """
    N = len(positions_cam)
    # Convert to homogeneous coordinates
    ones = np.ones((N, 1))
    pos_homo = np.hstack([positions_cam, ones])  # (N, 4)
    
    # Transform: p_world = T_world_cam @ p_cam
    pos_world_homo = (T_world_cam @ pos_homo.T).T  # (N, 4)
    
    return pos_world_homo[:, :3]


def calibrate_object_z_offset(
    mesh_points: np.ndarray,
    object_poses: np.ndarray,
    T_world_cam: np.ndarray
) -> float:
    """
    Calculate z-offset to place object bottom at z=0 using the first frame.
    
    The object is assumed to be sitting on the table in the first frame,
    so the lowest point of the mesh should be at z=0.
    
    Args:
        mesh_points: (P, 3) mesh points in object frame
        object_poses: (N, 4, 4) object poses in camera frame
        T_world_cam: 4x4 camera-to-world transform
        
    Returns:
        float: z-offset to add to all object positions
    """
    # Transform mesh to camera frame using first pose
    first_pose = object_poses[0]
    mesh_cam = transform_mesh_points(mesh_points, first_pose)
    
    # Transform to world frame
    mesh_world = transform_to_world(mesh_cam, T_world_cam)
    
    # Find the minimum z value (bottom of object)
    z_min = mesh_world[:, 2].min()
    
    # Offset to bring bottom to z=0
    z_offset = -z_min
    
    print(f"[INFO] Object z calibration:")
    print(f"       First frame mesh z_min = {z_min:.4f} m")
    print(f"       Applying z_offset = {z_offset:.4f} m")
    
    return z_offset


def apply_z_offset_to_poses(
    object_poses: np.ndarray,
    z_offset: float,
    T_world_cam: np.ndarray
) -> np.ndarray:
    """
    Apply z-offset to object poses.
    
    The offset is applied in world frame, so we need to transform it back
    to camera frame for each pose.
    
    Args:
        object_poses: (N, 4, 4) object poses in camera frame
        z_offset: z-offset to apply in world frame
        T_world_cam: 4x4 camera-to-world transform
        
    Returns:
        np.ndarray: (N, 4, 4) corrected object poses
    """
    # The z-offset is in world frame
    # World z-axis offset needs to be converted to camera frame offset
    # offset_cam = R_cam_world @ offset_world
    
    T_cam_world = np.linalg.inv(T_world_cam)
    R_cam_world = T_cam_world[:3, :3]
    
    # Offset in world frame: [0, 0, z_offset]
    offset_world = np.array([0, 0, z_offset])
    offset_cam = R_cam_world @ offset_world
    
    # Apply offset to all poses
    corrected_poses = object_poses.copy()
    corrected_poses[:, :3, 3] += offset_cam
    
    return corrected_poses


def apply_z_offset_to_trajectory(
    object_traj: np.ndarray,
    z_offset: float,
    T_world_cam: np.ndarray
) -> np.ndarray:
    """
    Apply z-offset to object trajectory (positions in camera frame).
    
    Args:
        object_traj: (N, 3) object positions in camera frame
        z_offset: z-offset to apply in world frame
        T_world_cam: 4x4 camera-to-world transform
        
    Returns:
        np.ndarray: (N, 3) corrected object positions
    """
    T_cam_world = np.linalg.inv(T_world_cam)
    R_cam_world = T_cam_world[:3, :3]
    
    offset_world = np.array([0, 0, z_offset])
    offset_cam = R_cam_world @ offset_world
    
    corrected_traj = object_traj.copy()
    corrected_traj += offset_cam
    
    return corrected_traj


def setup_3d_plot(
    hand_traj: np.ndarray,
    object_traj: np.ndarray = None,
    object_poses: np.ndarray = None,
    mesh_points: np.ndarray = None,
    T_world_cam: np.ndarray = None,
    title: str = "Hand + Object Trajectories",
    n_mesh_frames: int = 5
):
    """
    Setup 3D plot with trajectories and mesh. Returns fig, ax for further use.
    
    Args:
        hand_traj: (N, 6, 3) hand keypoints in world frame
        object_traj: (M, 3) object positions in camera frame (optional)
        object_poses: (M, 4, 4) object poses for mesh visualization (optional)
        mesh_points: (P, 3) sampled mesh points in object frame (optional)
        T_world_cam: 4x4 transform to convert object to world frame
        title: Plot title
        n_mesh_frames: Number of frames to show mesh at (evenly spaced)
        
    Returns:
        tuple: (fig, ax, mid, max_range) for animation use
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw table surface at z=0
    draw_table(ax, TABLE_SIZE_M, z_offset=0.0)
    
    # Filter out NaN frames from hand trajectory
    valid_frames = ~np.any(np.isnan(hand_traj[:, 0, :]), axis=1)
    hand_traj_valid = hand_traj[valid_frames]
    
    if len(hand_traj_valid) == 0:
        print("[WARN] No valid hand trajectory frames")
        return None, None, None, None
    
    # Plot hand trajectory for each keypoint
    for kp_idx in range(6):
        kp_traj = hand_traj_valid[:, kp_idx, :]  # (N, 3)
        
        # Plot trajectory line
        ax.plot(kp_traj[:, 0], kp_traj[:, 1], kp_traj[:, 2],
                color=KP_COLORS[kp_idx], linewidth=1.5, alpha=0.7,
                label=f"Hand: {KP_NAMES[kp_idx]}")
        
        # Plot start and end markers
        ax.scatter(kp_traj[0, 0], kp_traj[0, 1], kp_traj[0, 2],
                   color=KP_COLORS[kp_idx], s=100, marker='o', edgecolors='black')
        ax.scatter(kp_traj[-1, 0], kp_traj[-1, 1], kp_traj[-1, 2],
                   color=KP_COLORS[kp_idx], s=100, marker='s', edgecolors='black')
    
    # Draw skeleton lines from wrist to fingertips (at final frame)
    wrist_final = hand_traj_valid[-1, 0, :]
    if not np.any(np.isnan(wrist_final)):
        for kp_idx in range(1, 6):  # Fingertips (1-5)
            fingertip_final = hand_traj_valid[-1, kp_idx, :]
            if not np.any(np.isnan(fingertip_final)):
                ax.plot([wrist_final[0], fingertip_final[0]],
                       [wrist_final[1], fingertip_final[1]],
                       [wrist_final[2], fingertip_final[2]],
                       color=KP_COLORS[kp_idx], linewidth=2, alpha=0.7)
    
    # Plot object trajectory if available
    if object_traj is not None:
        obj_pos = object_traj
        
        # Transform to world frame if transform provided
        if T_world_cam is not None:
            obj_pos = transform_to_world(obj_pos, T_world_cam)
        
        # Plot object trajectory
        ax.plot(obj_pos[:, 0], obj_pos[:, 1], obj_pos[:, 2],
                color='#8800FF', linewidth=3, alpha=0.9, label='Object Center')
        
        # Start/end markers
        ax.scatter(obj_pos[0, 0], obj_pos[0, 1], obj_pos[0, 2],
                   color='#8800FF', s=200, marker='o', edgecolors='black', linewidths=2)
        ax.scatter(obj_pos[-1, 0], obj_pos[-1, 1], obj_pos[-1, 2],
                   color='#8800FF', s=200, marker='s', edgecolors='black', linewidths=2)
    
    # Plot mesh at selected frames if available
    if mesh_points is not None and object_poses is not None:
        n_poses = len(object_poses)
        # Select evenly spaced frames
        frame_indices = np.linspace(0, n_poses - 1, n_mesh_frames, dtype=int)
        
        # Color gradient from light to dark purple
        for i, frame_idx in enumerate(frame_indices):
            pose = object_poses[frame_idx]
            
            # Transform mesh points by pose (camera frame)
            mesh_cam = transform_mesh_points(mesh_points, pose)
            
            # Transform to world frame if extrinsics available
            if T_world_cam is not None:
                mesh_world = transform_to_world(mesh_cam, T_world_cam)
            else:
                mesh_world = mesh_cam
            
            # Color gradient: light purple (early) -> dark purple (late)
            alpha = 0.3 + 0.5 * (i / (n_mesh_frames - 1)) if n_mesh_frames > 1 else 0.6
            color_val = 0.9 - 0.5 * (i / (n_mesh_frames - 1)) if n_mesh_frames > 1 else 0.6
            
            label = f'Object (frame {frame_idx})' if i == 0 else None
            ax.scatter(mesh_world[:, 0], mesh_world[:, 1], mesh_world[:, 2],
                      c=[[color_val, 0.2, color_val]], s=5, alpha=alpha, label=label)
    
    # Set equal aspect ratio
    all_points = hand_traj_valid.reshape(-1, 3)
    if object_traj is not None:
        if T_world_cam is not None:
            obj_world = transform_to_world(object_traj, T_world_cam)
        else:
            obj_world = object_traj
        all_points = np.vstack([all_points, obj_world])
    
    # Remove NaNs for bounds calculation
    all_points = all_points[~np.any(np.isnan(all_points), axis=1)]
    
    max_range = np.ptp(all_points, axis=0).max() / 2.0
    mid = all_points.mean(axis=0)
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    # Labels and legend
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax, mid, max_range


def create_replay_video(
    hand_traj: np.ndarray,
    object_poses: np.ndarray = None,
    mesh_points: np.ndarray = None,
    T_world_cam: np.ndarray = None,
    video_path: Path = None,
    title: str = "Manipulation Replay",
    fps: int = 15,
    trail_length: int = 30
):
    """
    Create frame-by-frame replay video showing hand and object moving through time.
    
    Args:
        hand_traj: (N, 6, 3) hand keypoints in world frame
        object_poses: (M, 4, 4) object poses (optional)
        mesh_points: (P, 3) sampled mesh points in object frame (optional)
        T_world_cam: 4x4 transform to convert object to world frame
        video_path: output video path
        title: Plot title
        fps: frames per second
        trail_length: number of past frames to show as trail
    """
    import cv2
    import tempfile
    import shutil
    
    # Filter valid frames
    valid_mask = ~np.any(np.isnan(hand_traj[:, 0, :]), axis=1)
    n_frames = len(hand_traj)
    
    # Compute bounds from all valid data
    valid_hand = hand_traj[valid_mask].reshape(-1, 3)
    # Remove any remaining NaNs
    valid_hand = valid_hand[~np.any(np.isnan(valid_hand), axis=1)]
    all_points = valid_hand.copy()
    
    if object_poses is not None and mesh_points is not None:
        # Get all object positions for bounds (sample every 10th frame for speed)
        for i in range(0, len(object_poses), 10):
            pose = object_poses[i]
            mesh_cam = transform_mesh_points(mesh_points, pose)
            if T_world_cam is not None:
                mesh_world = transform_to_world(mesh_cam, T_world_cam)
            else:
                mesh_world = mesh_cam
            # Filter NaNs
            mesh_world = mesh_world[~np.any(np.isnan(mesh_world), axis=1)]
            if len(mesh_world) > 0:
                all_points = np.vstack([all_points, mesh_world])
    
    # Final NaN check
    all_points = all_points[~np.any(np.isnan(all_points), axis=1)]
    
    if len(all_points) == 0:
        print("[ERROR] No valid points for bounds calculation")
        return
    
    max_range = np.ptp(all_points, axis=0).max() / 2.0 * 1.2  # 20% padding
    mid = all_points.mean(axis=0)
    
    # Safety check
    if np.isnan(max_range) or np.any(np.isnan(mid)):
        print("[ERROR] Invalid bounds (NaN detected)")
        return
    
    # Define camera angles: (elevation, azimuth, name, position)
    # Position: (row, col) for 2x2 grid
    # azim=0: looking from +X toward origin, azim=90: looking from +Y, azim=180: from -X, azim=270: from -Y
    camera_angles = [
        (25, 225, "Side View (180°)", (0, 0)),  # Top left - main side view
        (90, 0, "Top Down (XY)", (0, 1)),       # Top right - looking down Z axis
        (0, 0, "YZ Plane (from +X)", (1, 0)),   # Bottom left - looking along X axis at YZ plane
        (0, 90, "XZ Plane (from +Y)", (1, 1)),  # Bottom right - looking along Y axis at XZ plane
    ]
    
    print(f"[INFO] Creating replay video with {n_frames} frames (2x2 grid view)...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    for frame_idx in range(n_frames):
        # Create 2x2 subplot grid
        fig = plt.figure(figsize=(20, 20))
        
        # Draw trajectory trails (past positions) - compute once per frame
        start_idx = max(0, frame_idx - trail_length)
        trails_data = []
        current_positions = []
        
        for kp_idx in range(6):
            trail = hand_traj[start_idx:frame_idx+1, kp_idx, :]
            valid = ~np.any(np.isnan(trail), axis=1)
            trail = trail[valid]
            trails_data.append(trail)
            
            current = hand_traj[frame_idx, kp_idx, :]
            if not np.any(np.isnan(current)):
                current_positions.append((kp_idx, current))
        
        # Object mesh at current frame
        object_mesh_world = None
        if object_poses is not None and mesh_points is not None and frame_idx < len(object_poses):
            pose = object_poses[frame_idx]
            mesh_cam = transform_mesh_points(mesh_points, pose)
            if T_world_cam is not None:
                object_mesh_world = transform_to_world(mesh_cam, T_world_cam)
            else:
                object_mesh_world = mesh_cam
        
        # Create 4 subplots
        for angle_idx, (elev, azim, angle_name, (row, col)) in enumerate(camera_angles):
            # Convert (row, col) to subplot index: 2 rows, 2 cols
            subplot_idx = row * 2 + col + 1
            ax = fig.add_subplot(2, 2, subplot_idx, projection='3d')
            
            # Draw table surface
            draw_table(ax, TABLE_SIZE_M, z_offset=0.0)
            
            # Draw trails
            for kp_idx, trail in enumerate(trails_data):
                if len(trail) > 1:
                    for j in range(1, len(trail)):
                        alpha = 0.2 + 0.6 * (j / len(trail))
                        ax.plot(trail[j-1:j+1, 0], trail[j-1:j+1, 1], trail[j-1:j+1, 2],
                               color=KP_COLORS[kp_idx], linewidth=1.5, alpha=alpha)
            
            # Draw current positions
            wrist_pos = None
            fingertip_positions = {}
            for kp_idx, pos in current_positions:
                ax.scatter(pos[0], pos[1], pos[2],
                          color=KP_COLORS[kp_idx], s=80, edgecolors='black',
                          label=KP_NAMES[kp_idx] if frame_idx == 0 and angle_idx == 0 else None)
                if kp_idx == 0:  # Wrist
                    wrist_pos = pos
                else:  # Fingertips (1-5)
                    fingertip_positions[kp_idx] = pos
            
            # Draw lines from wrist to each fingertip
            if wrist_pos is not None:
                for kp_idx, fingertip_pos in fingertip_positions.items():
                    ax.plot([wrist_pos[0], fingertip_pos[0]],
                           [wrist_pos[1], fingertip_pos[1]],
                           [wrist_pos[2], fingertip_pos[2]],
                           color=KP_COLORS[kp_idx], linewidth=2, alpha=0.7)
            
            # Draw object mesh
            if object_mesh_world is not None:
                ax.scatter(object_mesh_world[:, 0], object_mesh_world[:, 1], object_mesh_world[:, 2],
                          c='#8800FF', s=8, alpha=0.7, label='Object' if frame_idx == 0 and angle_idx == 0 else None)
            
            # Set view limits
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
            
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_zlabel('Z (m)', fontsize=10)
            ax.set_title(angle_name, fontsize=12, fontweight='bold')
            ax.view_init(elev=elev, azim=azim)
            ax.grid(True, alpha=0.3)
            
            if frame_idx == 0 and angle_idx == 0:
                ax.legend(loc='upper left', fontsize=7)
        
        # Add overall title
        fig.suptitle(f"{title} - Frame {frame_idx+1}/{n_frames}", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        
        # Save frame
        frame_path = temp_dir / f"{frame_idx:06d}.png"
        fig.savefig(str(frame_path), dpi=100)
        plt.close(fig)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"  Rendered {frame_idx + 1}/{n_frames} frames...")
    
    # Combine frames into video
    frame_files = sorted(temp_dir.glob("*.png"))
    if frame_files:
        first_img = cv2.imread(str(frame_files[0]))
        h, w = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        for fp in frame_files:
            img = cv2.imread(str(fp))
            out.write(img)
        out.release()
        print(f"[SAVED] Replay video -> {video_path}")
    
    # Cleanup
    shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Visualize hand + object 3D trajectories")
    parser.add_argument("--name", type=str, required=True,
                       help="Recording name (e.g., Jmanip1)")
    parser.add_argument("--camera", type=str, default="zed", choices=["zed", "realsense"],
                       help="Camera used for FoundationPose")
    parser.add_argument("--mesh", type=str, default=None,
                       help="Mesh name in FoundationPose/object/ (e.g., J, HEX)")
    parser.add_argument("--n_mesh_frames", type=int, default=5,
                       help="Number of frames to show mesh at")
    parser.add_argument("--n_mesh_points", type=int, default=500,
                       help="Number of points to sample from mesh")
    parser.add_argument("--no_interactive", action="store_true",
                       help="Skip interactive display (just save files)")
    parser.add_argument("--no_object", action="store_true",
                       help="Only plot hand trajectory (skip object)")
    parser.add_argument("--fps", type=int, default=15,
                       help="Video frame rate")
    parser.add_argument("--trail", type=int, default=30,
                       help="Trail length (number of past frames to show)")
    parser.add_argument("--hand_offset_x", type=float, default=-0.0133,
                       help="Offset to add to hand X coordinates (meters). Default: -0.0133")
    parser.add_argument("--hand_offset_y", type=float, default=0.0367,
                       help="Offset to add to hand Y coordinates (meters). Default: 0.0367")
    parser.add_argument("--hand_offset_z", type=float, default=0.0,
                       help="Offset to add to hand Z coordinates (meters). Default: 0.0")
    
    args = parser.parse_args()
    
    # Data paths
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / args.name
    calib_path = script_dir / "record" / "cam_calib"
    
    if not data_path.exists():
        print(f"[ERROR] Data folder not found: {data_path}")
        print(f"Available recordings:")
        data_root = script_dir / "data"
        if data_root.exists():
            for d in sorted(data_root.iterdir()):
                if d.is_dir():
                    print(f"  {d.name}")
        return
    
    # Load hand trajectory (in world frame)
    hand_traj = load_hand_trajectory(data_path)
    if hand_traj is None:
        print("[ERROR] Cannot load hand trajectory")
        return
    
    # Apply offsets to hand trajectory and save to _fixed.npy
    # Always loads from original hand_trajectory.npy, saves fixed version separately
    if args.hand_offset_x != 0 or args.hand_offset_y != 0 or args.hand_offset_z != 0:
        print(f"[INFO] Applying offsets to hand trajectory:")
        print(f"       X offset: {args.hand_offset_x:.3f} m")
        print(f"       Y offset: {args.hand_offset_y:.3f} m")
        print(f"       Z offset: {args.hand_offset_z:.3f} m")
        hand_traj = hand_traj.copy()  # Don't modify original in memory
        hand_traj[:, :, 0] += args.hand_offset_x
        hand_traj[:, :, 1] += args.hand_offset_y
        hand_traj[:, :, 2] += args.hand_offset_z
        
        # Save fixed version (separate from original so we can keep testing)
        fixed_path = data_path / "traj" / "hand_trajectory_fixed.npy"
        np.save(fixed_path, hand_traj)
        print(f"[SAVED] Fixed hand trajectory -> {fixed_path}")
        
        # Also save the full 21-keypoint version with offsets if it exists
        full_kp_path = data_path / "traj" / "righthand_3d_keypoints.npy"
        if full_kp_path.exists():
            full_kp = np.load(full_kp_path)
            full_kp_fixed = full_kp.copy()
            full_kp_fixed[:, :, 0] += args.hand_offset_x
            full_kp_fixed[:, :, 1] += args.hand_offset_y
            full_kp_fixed[:, :, 2] += args.hand_offset_z
            fixed_full_path = data_path / "traj" / "righthand_3d_keypoints_fixed.npy"
            np.save(fixed_full_path, full_kp_fixed)
            print(f"[SAVED] Fixed full keypoints -> {fixed_full_path}")
    
    # Load object trajectory, poses, and camera extrinsics
    object_traj = None
    object_poses = None
    mesh_points = None
    T_world_cam = None
    
    if not args.no_object:
        object_traj = load_object_trajectory(data_path, args.camera)
        
        # Load camera extrinsics to transform object to world frame
        cam_id = 1 if args.camera == "zed" else 2
        extrinsics_path = calib_path / f"cam{cam_id}_extrinsics.npy"
        if extrinsics_path.exists():
            T_world_cam = np.load(extrinsics_path)
            print(f"[INFO] Loaded camera extrinsics for transformation")
        else:
            print(f"[WARN] Camera extrinsics not found, object will be in camera frame")
        
        # Load mesh if specified
        if args.mesh:
            mesh_points = load_mesh_points(args.mesh, args.n_mesh_points)
            if mesh_points is not None:
                object_poses = load_object_poses(data_path, args.camera)
                
                # Calibrate z-offset: place object bottom at z=0 using first frame
                if object_poses is not None and T_world_cam is not None:
                    z_offset = calibrate_object_z_offset(mesh_points, object_poses, T_world_cam)
                    
                    # Apply z-offset to poses and trajectory
                    object_poses = apply_z_offset_to_poses(object_poses, z_offset, T_world_cam)
                    if object_traj is not None:
                        object_traj = apply_z_offset_to_trajectory(object_traj, z_offset, T_world_cam)
                    
                    # Save corrected data back to npy files
                    fp_folder = data_path / "traj" / "FP" / args.camera
                    
                    # Save corrected poses
                    corrected_poses_path = fp_folder / "object_poses.npy"
                    np.save(corrected_poses_path, object_poses)
                    print(f"[SAVED] Corrected object poses -> {corrected_poses_path}")
                    
                    # Save corrected trajectory
                    if object_traj is not None:
                        corrected_traj_path = fp_folder / "object_trajectory.npy"
                        np.save(corrected_traj_path, object_traj)
                        print(f"[SAVED] Corrected object trajectory -> {corrected_traj_path}")
    
    # Build title
    title = f"Trajectories: {args.name}"
    if object_traj is not None:
        title += f" (Object from {args.camera})"
    if args.mesh:
        title += f" [Mesh: {args.mesh}]"
    
    # Output paths
    traj_folder = data_path / "traj"
    traj_folder.mkdir(parents=True, exist_ok=True)
    png_path = traj_folder / "combined_trajectory_3d.png"
    video_path = traj_folder / "manipulation_replay.mp4"
    
    # Create static plot (shows all trajectories at once)
    fig, ax, mid, max_range = setup_3d_plot(
        hand_traj=hand_traj,
        object_traj=object_traj,
        object_poses=object_poses,
        mesh_points=mesh_points,
        T_world_cam=T_world_cam,
        title=title,
        n_mesh_frames=args.n_mesh_frames
    )
    
    if fig is None:
        return
    
    # Save static PNG (overview of all trajectories)
    fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
    print(f"[SAVED] Static overview PNG -> {png_path}")
    
    # Create frame-by-frame replay video
    create_replay_video(
        hand_traj=hand_traj,
        object_poses=object_poses,
        mesh_points=mesh_points,
        T_world_cam=T_world_cam,
        video_path=video_path,
        title=title,
        fps=args.fps,
        trail_length=args.trail
    )
    
    # Show interactive if not disabled
    if not args.no_interactive:
        print("[INFO] Opening interactive view (close window to exit)...")
        plt.show()
    
    plt.close(fig)


if __name__ == "__main__":
    main()

