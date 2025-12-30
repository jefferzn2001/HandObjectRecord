#!/usr/bin/env python3
"""
Step 2: Fast blue detection and mask generation (quick step, can run multiple times).

This script loads pre-computed SAM features, quickly detects blue objects,
displays results for manual confirmation/correction, and saves masks.

Usage:
    python sam_detect_blue.py --data_path data/test_wei_01
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import torch

from segment_anything import sam_model_registry, SamPredictor


def find_checkpoint_file(checkpoints_root: str) -> Optional[str]:
    """Find SAM checkpoint file in checkpoints directory."""
    path = Path(checkpoints_root)
    if not path.exists():
        print(f"[WARN] Checkpoints directory not found: {checkpoints_root}")
        return None

    ckpts = sorted(path.glob("*.pth"))
    if not ckpts:
        print(f"[WARN] No .pth checkpoint files under {checkpoints_root}")
        return None

    if len(ckpts) > 1:
        print(f"[INFO] Multiple checkpoints found, using: {ckpts[0]}")
    else:
        print(f"[INFO] Using checkpoint: {ckpts[0]}")
    return str(ckpts[0])


def load_sam_predictor(checkpoint_path: str, model_type: str) -> SamPredictor:
    """Load SAM model and return predictor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading SAM ({model_type}) from {checkpoint_path} onto {device}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)
    print("[INFO] SAM model ready.")
    return predictor


def load_saved_features(cam_root: Path, predictor: SamPredictor):
    """Load saved SAM features and set predictor state."""
    features_dir = cam_root / "sam_features"
    
    embedding_path = features_dir / "000000_embedding.npy"
    metadata_path = features_dir / "000000_metadata.npy"
    image_path = features_dir / "000000_image.png"
    
    if not embedding_path.exists() or not metadata_path.exists():
        return None, None
    
    # Load embedding and metadata
    image_embedding = torch.from_numpy(np.load(embedding_path)).to(predictor.model.device)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    
    # Load image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None, None
    
    # Set predictor state using saved features
    predictor.original_size = tuple(metadata['original_size'])
    predictor.input_size = tuple(metadata['input_size'])
    predictor.features = image_embedding
    predictor.is_image_set = True
    
    return image_bgr, metadata


def detect_largest_blue_object(image_bgr):
    """
    Automatically detect the largest blue object in the image.
    Returns five evenly distributed points within the blue region for SAM.
    """
    # Split BGR channels
    b_channel = image_bgr[:, :, 0].astype(np.float32)  # Blue channel
    g_channel = image_bgr[:, :, 1].astype(np.float32)  # Green channel
    r_channel = image_bgr[:, :, 2].astype(np.float32)  # Red channel
    
    # Blue detection criteria
    b_in_range = (b_channel >= 60) & (b_channel <= 150)
    b_minus_rg = b_channel - np.maximum(r_channel, g_channel)
    b_dominant = b_minus_rg > 20
    rg_low = (r_channel < b_channel) & (g_channel < b_channel) & (r_channel < 100) & (g_channel < 100)
    
    blue_mask = (b_in_range & b_dominant & rg_low).astype(np.uint8) * 255
    
    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
    
    if num_labels < 2:
        return None, None
    
    # Filter components
    h, w = image_bgr.shape[:2]
    image_area = h * w
    max_area = image_area / 6.0
    
    valid_indices = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        touches_edge = (x <= 0) or (x + width >= w) or (y <= 0) or (y + height >= h)
        too_large = area > max_area
        
        if not touches_edge and not too_large:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        return None, None
    
    # Find largest valid component
    valid_areas = [stats[i, cv2.CC_STAT_AREA] for i in valid_indices]
    largest_idx = valid_indices[np.argmax(valid_areas)]
    largest_mask = (labels == largest_idx).astype(np.uint8) * 255
    
    # Select 5 evenly distributed points
    blue_pixels = np.where(largest_mask > 0)
    blue_coords = np.column_stack((blue_pixels[1], blue_pixels[0]))
    
    x_min, x_max = blue_coords[:, 0].min(), blue_coords[:, 0].max()
    y_min, y_max = blue_coords[:, 1].min(), blue_coords[:, 1].max()
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    regions = [
        ((center_x - x_range * 0.2, center_x + x_range * 0.2),
         (center_y - y_range * 0.2, center_y + y_range * 0.2)),
        ((x_min, center_x), (y_min, center_y)),
        ((center_x, x_max), (y_min, center_y)),
        ((x_min, center_x), (center_y, y_max)),
        ((center_x, x_max), (center_y, y_max)),
    ]
    
    points = []
    for (x_low, x_high), (y_low, y_high) in regions:
        region_coords = blue_coords[
            (blue_coords[:, 0] >= x_low) & (blue_coords[:, 0] < x_high) &
            (blue_coords[:, 1] >= y_low) & (blue_coords[:, 1] < y_high)
        ]
        if len(region_coords) > 0:
            region_center = np.mean(region_coords, axis=0)
            closest = region_coords[np.argmin(np.sum((region_coords - region_center)**2, axis=1))]
            points.append([int(closest[0]), int(closest[1])])
    
    # Remove duplicates
    unique_points = []
    seen = set()
    for p in points:
        p_tuple = tuple(p)
        if p_tuple not in seen:
            seen.add(p_tuple)
            unique_points.append(p)
            if len(unique_points) >= 5:
                break
    
    return unique_points, largest_mask


def overlay_mask(image_bgr, mask_bool, points, alpha=0.5):
    """Overlay mask and click points on image."""
    vis = image_bgr.copy()
    if mask_bool is not None:
        mask_uint8 = (mask_bool.astype(np.uint8) * 255)
        mask_color = np.zeros_like(image_bgr)
        mask_color[:, :, 0] = mask_uint8
        vis = cv2.addWeighted(image_bgr, 1.0, mask_color, alpha, 0)
    
    for (x, y) in points:
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
    return vis


def process_camera(cam_name: str, cam_root: Path, predictor: SamPredictor):
    """Process one camera: detect blue, show result, allow manual correction."""
    # Load saved features
    image_bgr, metadata = load_saved_features(cam_root, predictor)
    if image_bgr is None:
        print(f"[ERROR] {cam_name}: Failed to load saved features. Run sam_extract_features.py first.")
        return None
    
    print(f"\n[PROCESSING] {cam_name}...")
    
    # Fast blue detection
    detected_points, blue_mask = detect_largest_blue_object(image_bgr)
    
    if detected_points is None or len(detected_points) == 0:
        print(f"[AUTO] {cam_name}: No blue object detected")
        # Fall back to manual mode
        return manual_mask_selection(cam_name, image_bgr, predictor)
    
    print(f"[AUTO] {cam_name}: Detected {len(detected_points)} points in blue region")
    
    # Generate mask using detected points
    points = np.array(detected_points)
    labels = np.ones(len(detected_points), dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    mask_result = masks[best_idx]
    
    print(f"[AUTO] {cam_name}: Generated mask with score {scores[best_idx]:.3f}")
    
    # Show result and allow manual correction
    vis = overlay_mask(image_bgr, mask_result, detected_points)
    window_name = f"Blue Detection - {cam_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    click_points = detected_points.copy()
    current_mask = mask_result.copy()
    current_vis = vis.copy()
    
    def recompute_mask():
        nonlocal current_mask, current_vis
        if len(click_points) == 0:
            current_mask = None
            current_vis = image_bgr.copy()
            return
        
        points = np.array(click_points)
        labels = np.ones(len(click_points), dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        current_mask = masks[best_idx]
        current_vis = overlay_mask(image_bgr, current_mask, click_points)
    
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append([x, y])
            recompute_mask()
        elif event == cv2.EVENT_RBUTTONDOWN and click_points:
            click_points.pop()
            recompute_mask()
    
    cv2.setMouseCallback(window_name, mouse_cb)
    print(f"[CONTROL] {cam_name}: left click=add point, right click=undo, 'c'=clear, 's'=save, 'n'=skip, 'q'=quit")
    
    while True:
        cv2.imshow(window_name, current_vis)
        key = cv2.waitKey(20) & 0xFF
        
        if key == ord('c'):
            click_points.clear()
            recompute_mask()
        elif key == ord('n'):
            print(f"[SKIP] {cam_name}: skipped")
            cv2.destroyWindow(window_name)
            return None
        elif key == ord('s'):
            if current_mask is None:
                print("[WARN] No mask to save. Click points first.")
                continue
            # Save mask
            mask_path = cam_root / "masks" / "000000_mask.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_uint8 = (current_mask.astype(np.uint8) * 255)
            cv2.imwrite(str(mask_path), mask_uint8)
            print(f"[SAVE] {cam_name}: Saved mask to {mask_path}")
            cv2.destroyWindow(window_name)
            return current_mask
        elif key == ord('q'):
            print(f"[QUIT] {cam_name}: aborted")
            cv2.destroyWindow(window_name)
            return None
    
    return None


def manual_mask_selection(cam_name: str, image_bgr: np.ndarray, predictor: SamPredictor):
    """Fallback to manual mask selection."""
    # For now, return None - user can manually click if needed
    print(f"[MANUAL] {cam_name}: Please use manual mode if needed")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Fast blue detection and mask generation (Step 2 - quick, can run multiple times)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to recording folder (e.g., data/test_wei_01)",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint (.pth). Defaults to first file found in --checkpoints-root.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=str,
        default="checkpoints",
        help="Directory to search for SAM checkpoints if --sam-checkpoint is not set.",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model size to load.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"[ERROR] Data path not found: {data_path}")
        return

    # Find checkpoint
    sam_checkpoint_path = args.sam_checkpoint or find_checkpoint_file(args.checkpoints_root)
    if not sam_checkpoint_path:
        print("[ERROR] No SAM checkpoint found.")
        return

    # Load SAM model
    predictor = load_sam_predictor(sam_checkpoint_path, args.sam_model_type)

    # Process each camera
    cameras = [
        ("RealSense", data_path / "realsense"),
        ("ZED", data_path / "zed"),
    ]
    
    print("\n" + "="*60)
    print("STEP 2: Fast blue detection and mask generation")
    print("="*60)
    
    for cam_name, cam_root in cameras:
        if not cam_root.exists():
            print(f"[SKIP] {cam_name}: camera folder not found")
            continue
        process_camera(cam_name, cam_root, predictor)
    
    cv2.destroyAllWindows()
    print("\n[INFO] Processing complete.")


if __name__ == "__main__":
    main()

