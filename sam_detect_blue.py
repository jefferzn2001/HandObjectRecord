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
    Automatically detect the largest blue object in the image using global search.
    Uses histogram equalization for brightness normalization.
    Returns five evenly distributed points within the blue region for SAM.
    """
    # Apply histogram equalization for brightness normalization
    # Convert to LAB color space for better brightness separation
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel_lab = cv2.split(image_lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_eq = clahe.apply(l_channel)
    
    # Merge back and convert to BGR
    image_lab_eq = cv2.merge([l_channel_eq, a_channel, b_channel_lab])
    image_bgr_eq = cv2.cvtColor(image_lab_eq, cv2.COLOR_LAB2BGR)
    
    # Convert to HSV for better color-based detection
    image_hsv = cv2.cvtColor(image_bgr_eq, cv2.COLOR_BGR2HSV)
    h_channel = image_hsv[:, :, 0].astype(np.float32)
    s_channel = image_hsv[:, :, 1].astype(np.float32)
    v_channel = image_hsv[:, :, 2].astype(np.float32)
    
    # Also use BGR for additional blue detection
    b_channel = image_bgr_eq[:, :, 0].astype(np.float32)
    g_channel = image_bgr_eq[:, :, 1].astype(np.float32)
    r_channel = image_bgr_eq[:, :, 2].astype(np.float32)
    
    # Calculate "blueness" score for each pixel (global, not center-biased)
    # Based on user examples: RGB(0,26,72) and RGB(38,16,29)
    # These are dark blues with low overall brightness
    
    # HSV: For dark blues, hue can vary, but we want to catch blue tones
    # Dark blue like RGB(0,26,72) has low value, so we need to lower thresholds
    h_blue = ((h_channel >= 100) & (h_channel <= 130)).astype(np.float32)
    # Lower saturation threshold for dark blues (they may have lower saturation)
    s_acceptable = (s_channel > 20).astype(np.float32)
    # Lower value threshold to catch dark blues (value can be very low)
    v_acceptable = (v_channel > 10).astype(np.float32)
    
    # BGR-based blue detection for dark blues
    # For RGB(0,26,72) -> BGR(72,26,0): B=72, G=26, R=0 (B >> R, B >> G)
    # For RGB(38,16,29) -> BGR(29,16,38): B=29, G=16, R=38 (B > G, but R > B)
    # So we need to handle cases where B is not always dominant but still blue
    
    # Primary case: B is clearly dominant (like RGB(0,26,72))
    b_strongly_dominant = (b_channel - np.maximum(r_channel, g_channel) > 15).astype(np.float32)
    
    # Secondary case: B is higher than G, and R is not too high (like RGB(38,16,29))
    # B > G and B is at least moderate, R is not dominant
    b_moderate = ((b_channel > g_channel) & (b_channel > 20) & 
                  (r_channel < 80)).astype(np.float32)
    
    # R and G should be relatively low for blue
    rg_low = ((r_channel < 80) & (g_channel < 80)).astype(np.float32)
    
    # Combined: B should be at least moderate for blue
    b_minimum = (b_channel > 15).astype(np.float32)
    
    # Combined blueness score (weighted combination for dark blues)
    blueness_score = (
        h_blue * 0.3 +  # HSV hue (may be less reliable for very dark colors)
        s_acceptable * 0.15 +  # Acceptable saturation
        v_acceptable * 0.1 +  # Acceptable brightness (can be very low)
        b_strongly_dominant * 0.25 +  # B strongly dominant (primary case)
        b_moderate * 0.1 +  # B moderate but still blue (secondary case)
        rg_low * 0.15 +  # Low red/green
        b_minimum * 0.05  # B channel minimum threshold
    )
    
    # Lower threshold to catch dark blues
    blue_mask = (blueness_score > 0.4).astype(np.uint8) * 255
    
    # Apply morphological operations (dilation then erosion for better connectivity)
    # Use a larger kernel for better noise removal and connectivity
    kernel = np.ones((7, 7), np.uint8)
    
    # First: Dilation (expand blue regions to connect nearby pixels)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=2)
    
    # Second: Erosion (shrink back to remove noise and smooth boundaries)
    blue_mask = cv2.erode(blue_mask, kernel, iterations=2)
    
    # Optional: Additional closing to fill small holes
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
    
    if num_labels < 2:
        return None, None
    
    # Filter components
    h, w = image_bgr.shape[:2]
    image_area = h * w
    min_area = image_area / 500.0  # Blue object should be at least 1/500 of image area
    max_area = image_area / 10.0   # Blue object should be at most 1/10 of image area
    
    valid_indices = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        touches_edge = (x <= 0) or (x + width >= w) or (y <= 0) or (y + height >= h)
        too_small = area < min_area
        too_large = area > max_area
        
        if not touches_edge and not too_small and not too_large:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        return None, None
    
    # Find the bluest valid component (global search, not biased by size or position)
    # Calculate average blueness score for each valid component
    component_scores = []
    for i in valid_indices:
        component_mask = (labels == i).astype(np.bool_)
        # Get average blueness score for this component
        avg_blueness = np.mean(blueness_score[component_mask])
        component_scores.append((i, avg_blueness))
    
    # Select the component with highest average blueness
    bluest_idx = max(component_scores, key=lambda x: x[1])[0]
    bluest_mask = (labels == bluest_idx).astype(np.uint8) * 255
    
    # Select 5 evenly distributed points within the bluest region
    blue_pixels = np.where(bluest_mask > 0)
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
    
    return unique_points, bluest_mask


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
        return manual_mask_selection(cam_name, image_bgr, predictor, cam_root)
    
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


def manual_mask_selection(cam_name: str, image_bgr: np.ndarray, predictor: SamPredictor, cam_root: Path):
    """Manual mask selection when auto-detection fails."""
    print(f"[MANUAL] {cam_name}: Auto-detection failed. Please click points to select the object.")
    
    window_name = f"Manual Selection - {cam_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    click_points = []
    current_mask = None
    current_vis = image_bgr.copy()
    
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
    
    # Add instruction text
    instruction = "Click points on the object. Right-click to undo. 's'=save, 'c'=clear, 'n'=skip, 'q'=quit"
    print(f"[CONTROL] {cam_name}: {instruction}")
    
    while True:
        # Draw instruction on image
        vis_with_text = current_vis.copy()
        cv2.putText(vis_with_text, "Click on object (Left=add, Right=undo)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_with_text, "s=save, c=clear, n=skip, q=quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, vis_with_text)
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


def main():
    parser = argparse.ArgumentParser(
        description="Fast blue detection and mask generation (Step 2 - quick, can run multiple times)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Parent directory path (e.g., 'data/test_wei_02'). Will process the highest numbered subdirectory (1, 2, 3, ...)",
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

    root = Path(__file__).parent
    
    # Get parent directory path
    parent_path = Path(args.data_path)
    if not parent_path.is_absolute():
        parent_path = root / parent_path
    
    if not parent_path.exists():
        print(f"[ERROR] Parent directory not found: {parent_path}")
        return
    
    # Find all matching subdirectories (V####PERSON####SEQ######## format)
    existing_dirs = []
    for d in parent_path.iterdir():
        if d.is_dir() and d.name.startswith("V") and "PERSON" in d.name and "SEQ" in d.name:
            # Skip temporary directories
            if d.name.endswith("_TEMP"):
                continue
            try:
                # Extract sequence number from V####PERSON####SEQ########
                # Format: V####PERSON####SEQ########
                if "SEQ" in d.name:
                    seq_part = d.name.split("SEQ")[1]
                    seq_num = int(seq_part)
                    existing_dirs.append((d, seq_num))
            except (ValueError, IndexError):
                continue
    
    if not existing_dirs:
        print(f"[ERROR] No numbered subdirectories found in {parent_path}")
        print(f"[INFO] Please run record.py first to create a recording.")
        return
    
    # Sort by sequence number
    existing_dirs.sort(key=lambda x: x[1])
    print(f"[INFO] Found {len(existing_dirs)} directories to process")

    # Find checkpoint
    sam_checkpoint_path = args.sam_checkpoint or find_checkpoint_file(args.checkpoints_root)
    if not sam_checkpoint_path:
        print("[ERROR] No SAM checkpoint found.")
        return

    # Load SAM model (once for all directories)
    predictor = load_sam_predictor(sam_checkpoint_path, args.sam_model_type)

    # Process each directory
    print("\n" + "="*60)
    print("STEP 2: Fast blue detection and mask generation")
    print("="*60)
    
    for idx, (data_path, seq_num) in enumerate(existing_dirs, 1):
        print(f"\n[{idx}/{len(existing_dirs)}] Processing directory: {data_path.name}")
        
        # Process each camera
        cameras = [
            ("RealSense", data_path / "realsense"),
            ("ZED", data_path / "zed"),
        ]
        
        for cam_name, cam_root in cameras:
            if not cam_root.exists():
                print(f"[SKIP] {cam_name}: camera folder not found")
                continue
            process_camera(cam_name, cam_root, predictor)
    
    cv2.destroyAllWindows()
    print(f"\n[INFO] Processing complete for {len(existing_dirs)} directories.")


if __name__ == "__main__":
    main()

