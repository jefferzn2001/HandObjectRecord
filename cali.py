#!/usr/bin/env python3
"""
Calibration script for HaMeR hand offset.

Visualizes the index fingertip position at a specific frame where
the finger was placed at the origin (0,0,0) to determine the
calibration offset needed.

Usage:
    python cali.py --name Jcali --frame 328
    python cali.py --name Jcali --frame 328 --mesh J
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

# Try to import trimesh for mesh loading
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# FoundationPose mesh directory
FP_MESH_DIR = Path("/home/jeff/Desktop/FoundationPose/object")

# Table dimensions: 36" x 36" in meters
TABLE_SIZE_INCHES = 36
TABLE_SIZE_M = TABLE_SIZE_INCHES * 0.0254  # ~0.9144 meters

# Keypoint names (index 2 is Index fingertip in the 6-keypoint format)
KP_NAMES = ["Wrist", "Thumb", "Index", "Middle", "Ring", "Pinky"]
KP_COLORS = ["#00FF00", "#FF0000", "#FF8800", "#FFFF00", "#00FFFF", "#FF00FF"]


def load_hand_trajectory(data_path: Path) -> np.ndarray:
    """Load hand trajectory from HaMeR output."""
    traj_path = data_path / "traj" / "hand_trajectory.npy"
    if not traj_path.exists():
        print(f"[ERROR] Hand trajectory not found: {traj_path}")
        return None
    
    traj = np.load(traj_path)
    print(f"[INFO] Loaded hand trajectory: {traj.shape} (frames, keypoints, xyz)")
    return traj


def load_object_trajectory(data_path: Path, camera: str = "zed") -> np.ndarray:
    """Load object trajectory from FoundationPose."""
    traj_path = data_path / "traj" / "FP" / camera / "object_trajectory.npy"
    if not traj_path.exists():
        print(f"[WARN] Object trajectory not found: {traj_path}")
        return None
    
    traj = np.load(traj_path)
    print(f"[INFO] Loaded object trajectory: {traj.shape}")
    return traj


def load_object_poses(data_path: Path, camera: str = "zed") -> np.ndarray:
    """Load object poses from FoundationPose."""
    poses_path = data_path / "traj" / "FP" / camera / "object_poses.npy"
    if not poses_path.exists():
        return None
    return np.load(poses_path)


def load_mesh_points(mesh_name: str, n_points: int = 500) -> np.ndarray:
    """Load mesh and sample points from surface."""
    if not HAS_TRIMESH:
        print("[WARN] trimesh not installed")
        return None
    
    mesh_dir = FP_MESH_DIR / mesh_name
    if not mesh_dir.exists():
        print(f"[WARN] Mesh directory not found: {mesh_dir}")
        return None
    
    obj_files = list(mesh_dir.glob("*.obj"))
    if not obj_files:
        return None
    
    mesh = trimesh.load(str(obj_files[0]))
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def transform_mesh_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Transform mesh points by a 4x4 pose matrix."""
    N = len(points)
    ones = np.ones((N, 1))
    points_homo = np.hstack([points, ones])
    transformed = (pose @ points_homo.T).T
    return transformed[:, :3]


def transform_to_world(positions_cam: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """Transform positions from camera frame to world frame."""
    if positions_cam.ndim == 1:
        positions_cam = positions_cam.reshape(1, -1)
    N = len(positions_cam)
    ones = np.ones((N, 1))
    pos_homo = np.hstack([positions_cam, ones])
    pos_world_homo = (T_world_cam @ pos_homo.T).T
    return pos_world_homo[:, :3]


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
    
    # Draw table border with inch labels
    ax.plot([-half, half, half, -half, -half], 
            [-half, -half, half, half, -half],
            [z_offset, z_offset, z_offset, z_offset, z_offset],
            color='saddlebrown', linewidth=3)
    
    # Add dimension labels (in inches)
    ax.text(0, -half - 0.05, z_offset, f'{TABLE_SIZE_INCHES}"', 
            fontsize=10, ha='center', color='saddlebrown')
    ax.text(-half - 0.05, 0, z_offset, f'{TABLE_SIZE_INCHES}"', 
            fontsize=10, ha='center', va='center', rotation=90, color='saddlebrown')


def main():
    parser = argparse.ArgumentParser(description="Calibrate HaMeR hand offset")
    parser.add_argument("--name", type=str, required=True,
                       help="Recording name (e.g., Jcali)")
    parser.add_argument("--frame", type=int, required=True,
                       help="Frame number where finger is at origin (0-indexed)")
    parser.add_argument("--camera", type=str, default="zed",
                       help="Camera used for FoundationPose")
    parser.add_argument("--mesh", type=str, default=None,
                       help="Mesh name to visualize object")
    parser.add_argument("--n_mesh_points", type=int, default=300,
                       help="Number of mesh points to sample")
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / args.name
    calib_path = script_dir / "record" / "cam_calib"
    
    if not data_path.exists():
        print(f"[ERROR] Data folder not found: {data_path}")
        return
    
    # Load hand trajectory
    hand_traj = load_hand_trajectory(data_path)
    if hand_traj is None:
        return
    
    n_frames = hand_traj.shape[0]
    if args.frame < 0 or args.frame >= n_frames:
        print(f"[ERROR] Frame {args.frame} out of range (0 to {n_frames-1})")
        return
    
    # Get hand keypoints at specified frame
    hand_frame = hand_traj[args.frame]  # (6, 3)
    
    # Index fingertip is keypoint index 2
    index_pos = hand_frame[2]  # (3,)
    wrist_pos = hand_frame[0]  # (3,)
    
    print(f"\n{'='*60}")
    print(f"CALIBRATION ANALYSIS - Frame {args.frame}")
    print(f"{'='*60}")
    print(f"\nHand Keypoints at Frame {args.frame}:")
    print(f"-" * 40)
    for i, name in enumerate(KP_NAMES):
        pos = hand_frame[i]
        if np.any(np.isnan(pos)):
            print(f"  {name:8s}: NaN (not detected)")
        else:
            print(f"  {name:8s}: X={pos[0]:+.4f}, Y={pos[1]:+.4f}, Z={pos[2]:+.4f} m")
    
    print(f"\n{'='*60}")
    print(f"INDEX FINGERTIP (calibration point):")
    print(f"{'='*60}")
    if np.any(np.isnan(index_pos)):
        print(f"  [ERROR] Index finger not detected at frame {args.frame}")
        return
    
    print(f"  Current position: ({index_pos[0]:+.4f}, {index_pos[1]:+.4f}, {index_pos[2]:+.4f}) m")
    print(f"  Expected position: (0.0000, 0.0000, 0.0000) m")
    print(f"\n  OFFSET NEEDED (subtract this from HaMeR data):")
    print(f"    --hand_offset_x {-index_pos[0]:.4f}")
    print(f"    --hand_offset_y {-index_pos[1]:.4f}")
    print(f"    --hand_offset_z {-index_pos[2]:.4f}")
    
    # Load object data if available
    object_traj = load_object_trajectory(data_path, args.camera)
    object_poses = load_object_poses(data_path, args.camera)
    mesh_points = None
    T_world_cam = None
    
    # Load camera extrinsics
    cam_id = 1 if args.camera == "zed" else 2
    extrinsics_path = calib_path / f"cam{cam_id}_extrinsics.npy"
    if extrinsics_path.exists():
        T_world_cam = np.load(extrinsics_path)
    
    if object_traj is not None and args.frame < len(object_traj):
        obj_pos_cam = object_traj[args.frame]
        if T_world_cam is not None:
            obj_pos_world = transform_to_world(obj_pos_cam.reshape(1, 3), T_world_cam)[0]
        else:
            obj_pos_world = obj_pos_cam
        
        print(f"\n{'='*60}")
        print(f"OBJECT POSITION at Frame {args.frame}:")
        print(f"{'='*60}")
        print(f"  Camera frame: ({obj_pos_cam[0]:+.4f}, {obj_pos_cam[1]:+.4f}, {obj_pos_cam[2]:+.4f}) m")
        print(f"  World frame:  ({obj_pos_world[0]:+.4f}, {obj_pos_world[1]:+.4f}, {obj_pos_world[2]:+.4f}) m")
    
    # Load mesh if specified
    if args.mesh:
        mesh_points = load_mesh_points(args.mesh, args.n_mesh_points)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')  # Top view
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')  # Side view XZ
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')  # Side view YZ
    
    axes = [ax1, ax2, ax3, ax4]
    titles = ["3D View", "Top Down (XY)", "Side View (XZ)", "Front View (YZ)"]
    views = [(25, 225), (90, 0), (0, 0), (0, 90)]
    
    for ax, title, (elev, azim) in zip(axes, titles, views):
        # Draw table
        draw_table(ax, TABLE_SIZE_M, z_offset=0.0)
        
        # Draw origin
        ax.scatter([0], [0], [0], color='black', s=200, marker='x', linewidths=3, 
                   label='Origin (0,0,0)', zorder=10)
        
        # Draw coordinate axes
        axis_len = 0.15
        ax.quiver(0, 0, 0, axis_len, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
        ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
        ax.quiver(0, 0, 0, 0, 0, axis_len, color='blue', arrow_length_ratio=0.1, linewidth=2)
        ax.text(axis_len + 0.02, 0, 0, 'X', color='red', fontsize=10)
        ax.text(0, axis_len + 0.02, 0, 'Y', color='green', fontsize=10)
        ax.text(0, 0, axis_len + 0.02, 'Z', color='blue', fontsize=10)
        
        # Draw all hand keypoints
        for kp_idx in range(6):
            pos = hand_frame[kp_idx]
            if not np.any(np.isnan(pos)):
                marker_size = 200 if kp_idx == 2 else 80  # Larger for index finger
                ax.scatter(pos[0], pos[1], pos[2], color=KP_COLORS[kp_idx], 
                          s=marker_size, edgecolors='black', linewidths=1,
                          label=f'{KP_NAMES[kp_idx]}' if ax == ax1 else None)
        
        # Draw skeleton lines from wrist to fingertips
        if not np.any(np.isnan(wrist_pos)):
            for kp_idx in range(1, 6):
                fingertip = hand_frame[kp_idx]
                if not np.any(np.isnan(fingertip)):
                    ax.plot([wrist_pos[0], fingertip[0]],
                           [wrist_pos[1], fingertip[1]],
                           [wrist_pos[2], fingertip[2]],
                           color=KP_COLORS[kp_idx], linewidth=2, alpha=0.7)
        
        # Draw line from index finger to origin (error vector)
        if not np.any(np.isnan(index_pos)):
            ax.plot([index_pos[0], 0], [index_pos[1], 0], [index_pos[2], 0],
                   color='red', linewidth=2, linestyle='--', alpha=0.8,
                   label='Offset Error' if ax == ax1 else None)
        
        # Draw object mesh if available
        if mesh_points is not None and object_poses is not None and args.frame < len(object_poses):
            pose = object_poses[args.frame]
            mesh_cam = transform_mesh_points(mesh_points, pose)
            if T_world_cam is not None:
                mesh_world = transform_to_world(mesh_cam, T_world_cam)
            else:
                mesh_world = mesh_cam
            ax.scatter(mesh_world[:, 0], mesh_world[:, 1], mesh_world[:, 2],
                      c='#8800FF', s=8, alpha=0.6, label='Object' if ax == ax1 else None)
        
        # Set equal aspect ratio
        half_table = TABLE_SIZE_M / 2 * 1.2  # Slightly larger than table
        ax.set_xlim(-half_table, half_table)
        ax.set_ylim(-half_table, half_table)
        ax.set_zlim(-0.1, half_table)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)
        
        if ax == ax1:
            ax.legend(loc='upper left', fontsize=8)
    
    # Add text summary
    summary = f"Frame {args.frame} | Index Finger: ({index_pos[0]:+.4f}, {index_pos[1]:+.4f}, {index_pos[2]:+.4f}) m"
    summary += f"\nOffset needed: X={-index_pos[0]:.4f}, Y={-index_pos[1]:.4f}, Z={-index_pos[2]:.4f} m"
    fig.suptitle(f"Calibration: {args.name}\n{summary}", fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save figure
    out_path = data_path / "traj" / f"calibration_frame{args.frame}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Calibration plot -> {out_path}")
    
    plt.show()


if __name__ == "__main__":
    main()

