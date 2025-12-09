import os
import argparse
import cv2
import numpy as np
from pathlib import Path

from segment_anything import sam_model_registry, SamPredictor

# Global state for mouse interaction
click_points = []
click_labels = []
current_image = None
current_image_vis = None
predictor = None
mask_result = None


def mouse_callback(event, x, y, flags, param):
    global click_points, click_labels, current_image_vis, mask_result

    if event == cv2.EVENT_LBUTTONDOWN:
        # Left click = foreground point
        click_points.append([x, y])
        click_labels.append(1)  # 1 = foreground in SAM
        print(f"Added foreground point at ({x}, {y}), total points: {len(click_points)}")

        # Visualize click
        cv2.circle(current_image_vis, (x, y), 4, (0, 0, 255), -1)

        # Update mask prediction
        if len(click_points) > 0:
            mask_result = run_sam_on_points(current_image, click_points, click_labels, predictor)
            overlay = overlay_mask(current_image, mask_result)
            current_image_vis = overlay


def run_sam_on_points(image_bgr, points, labels, predictor):
    """
    Runs SAM with the given click points and returns a single best mask (HxW bool array).
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    point_coords = np.array(points)
    point_labels = np.array(labels)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )

    # Pick the mask with highest score
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]
    print(f"SAM produced {len(scores)} masks, best score: {scores[best_idx]:.3f}")
    return best_mask  # bool mask


def overlay_mask(image_bgr, mask_bool, alpha=0.5):
    """
    Overlays a semi-transparent mask on the image for visualization.
    """
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    mask_color = np.zeros_like(image_bgr)
    # Color the mask region (e.g., blue)
    mask_color[:, :, 0] = mask_uint8  # B channel

    overlay = cv2.addWeighted(image_bgr, 1.0, mask_color, alpha, 0)
    # Also draw the click points again
    for (x, y) in click_points:
        cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)
    return overlay


def find_latest_data_directory(data_root="data"):
    """
    Finds the latest data directory in the data folder.
    
    Args:
        data_root (str): Root directory containing data subdirectories.
        
    Returns:
        str: Path to the latest data directory, or None if not found.
    """
    data_path = Path(data_root)
    if not data_path.exists():
        print(f"[ERROR] Data directory not found: {data_root}")
        return None
    
    # Get all subdirectories
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"[ERROR] No subdirectories found in {data_root}")
        return None
    
    # Sort by modification time (most recent first)
    subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_dir = subdirs[0]
    print(f"[INFO] Using latest data directory: {latest_dir}")
    return str(latest_dir)


def find_checkpoint_file(checkpoints_root="checkpoints"):
    """
    Finds the SAM checkpoint file in the checkpoints directory.
    
    Args:
        checkpoints_root (str): Root directory containing checkpoint files.
        
    Returns:
        str: Path to the checkpoint file, or None if not found.
    """
    checkpoints_path = Path(checkpoints_root)
    if not checkpoints_path.exists():
        print(f"[ERROR] Checkpoints directory not found: {checkpoints_root}")
        return None
    
    # Look for .pth files
    checkpoint_files = list(checkpoints_path.glob("*.pth"))
    if not checkpoint_files:
        print(f"[ERROR] No .pth checkpoint files found in {checkpoints_root}")
        return None
    
    # If multiple, use the first one (or could sort by modification time)
    checkpoint_file = checkpoint_files[0]
    if len(checkpoint_files) > 1:
        print(f"[WARN] Multiple checkpoint files found, using: {checkpoint_file}")
    else:
        print(f"[INFO] Using checkpoint: {checkpoint_file}")
    return str(checkpoint_file)


def find_first_rgb_image(folder):
    """
    Finds the first RGB image inside folder/rgb.
    Returns (image_path or None).
    """
    rgb_dir = os.path.join(folder, "rgb")
    if not os.path.isdir(rgb_dir):
        print(f"[WARN] No 'rgb' folder under {folder}, skipping.")
        return None

    files = sorted([
        f for f in os.listdir(rgb_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if not files:
        print(f"[WARN] No RGB images found in {rgb_dir}, skipping.")
        return None

    first_img_path = os.path.join(rgb_dir, files[0])
    return first_img_path


def process_camera(camera_root, cam_name, predictor):
    global click_points, click_labels, current_image, current_image_vis, mask_result

    print(f"\n=== Processing camera: {cam_name} ===")
    img_path = find_first_rgb_image(camera_root)
    if img_path is None:
        return

    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f"[ERROR] Failed to load image: {img_path}")
        return

    current_image = image_bgr.copy()
    current_image_vis = image_bgr.copy()
    click_points = []
    click_labels = []
    mask_result = None

    window_name = f"SAM2D - {cam_name} (click to add FG points, 's' to save, 'n' next, 'q' quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, current_image_vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("Quitting all.")
            cv2.destroyAllWindows()
            return "quit"

        if key == ord('n'):
            print(f"Skipping save for {cam_name}. Moving on.")
            break

        if key == ord('s'):
            if mask_result is None:
                print("No mask yet. Click at least one point first.")
                continue

            # Save mask
            rgb_dir = os.path.join(camera_root, "rgb")
            base_name = os.path.basename(img_path)
            name_no_ext, _ = os.path.splitext(base_name)

            masks_dir = os.path.join(camera_root, "masks")
            os.makedirs(masks_dir, exist_ok=True)

            mask_path = os.path.join(masks_dir, f"{name_no_ext}_mask.png")
            mask_uint8 = (mask_result.astype(np.uint8) * 255)
            cv2.imwrite(mask_path, mask_uint8)
            print(f"Saved mask to: {mask_path}")

            break

    cv2.destroyWindow(window_name)
    return "continue"


def main():
    parser = argparse.ArgumentParser(description="Interactive SAM2D mask generator for ZED + RealSense.")
    parser.add_argument(
        "--record_root",
        type=str,
        default=None,
        help="Root folder of the recording that contains RealSense/ and ZED/ subfolders. If not provided, uses latest in data/.",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="Path to sam_vit_*.pth checkpoint. If not provided, uses latest in checkpoints/.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory containing data subdirectories (default: 'data').",
    )
    parser.add_argument(
        "--checkpoints_root",
        type=str,
        default="checkpoints",
        help="Root directory containing checkpoint files (default: 'checkpoints').",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type.",
    )
    args = parser.parse_args()

    # Auto-detect record_root if not provided
    if args.record_root is None:
        record_root = find_latest_data_directory(args.data_root)
        if record_root is None:
            print("[ERROR] Could not find latest data directory. Please specify --record_root.")
            return
    else:
        record_root = args.record_root

    # Auto-detect checkpoint if not provided
    if args.sam_checkpoint is None:
        sam_checkpoint = find_checkpoint_file(args.checkpoints_root)
        if sam_checkpoint is None:
            print("[ERROR] Could not find checkpoint file. Please specify --sam_checkpoint.")
            return
    else:
        sam_checkpoint = args.sam_checkpoint

    # Load SAM model
    print("Loading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=sam_checkpoint)
    sam.to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    global predictor
    predictor = SamPredictor(sam)
    print("SAM model loaded.")

    # Try both uppercase and lowercase camera folder names
    camera_variants = [
        ("RealSense", ["RealSense", "realsense"]),
        ("ZED", ["ZED", "zed"]),
    ]

    cams = []
    for cam_name, variants in camera_variants:
        found = False
        for variant in variants:
            cam_root = os.path.join(record_root, variant)
            if os.path.isdir(cam_root):
                cams.append((cam_name, cam_root))
                found = True
                break
        if not found:
            print(f"[INFO] Camera folder not found for {cam_name} (tried: {variants})")

    if not cams:
        print("[ERROR] No camera folders found. Expected RealSense/ or realsense/ and ZED/ or zed/ subfolders.")
        return

    for cam_name, cam_root in cams:
        status = process_camera(cam_root, cam_name, predictor)
        if status == "quit":
            break

    print("Done.")


if __name__ == "__main__":
    main()
