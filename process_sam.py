#!/usr/bin/env python3
"""
SAM (Segment Anything Model) processing for recorded data.

This script processes the first frame of a recording to generate masks
for both RealSense and ZED cameras using interactive SAM.

Usage:
    python process_sam.py --data_path data/test_wei_01
    python process_sam.py --data_path data/test_wei_01 --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

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


def detect_largest_blue_object(image_bgr):
    """
    Automatically detect the largest blue object in the image.
    Returns three evenly distributed points within the blue region for SAM.
    
    Blue detection logic: Blue has high B channel (60-150) and low R, G channels.
    Uses difference method: B should be significantly higher than R and G.
    
    Args:
        image_bgr: Input image in BGR format (H, W, 3)
    
    Returns:
        tuple: (points_list, mask) where points_list is [(x1,y1), (x2,y2), (x3,y3)] and mask is binary mask
               Returns (None, None) if no blue object found
    """
    # Split BGR channels
    # Note: OpenCV uses BGR format, so:
    # image_bgr[:,:,0] = B (blue)
    # image_bgr[:,:,1] = G (green)  
    # image_bgr[:,:,2] = R (red)
    b_channel = image_bgr[:, :, 0].astype(np.float32)  # Blue channel
    g_channel = image_bgr[:, :, 1].astype(np.float32)  # Green channel
    r_channel = image_bgr[:, :, 2].astype(np.float32)  # Red channel
    
    # Blue detection criteria:
    # 1. B channel should be in range 60-150
    # 2. B should be significantly higher than R and G
    # 3. R and G should be relatively low
    
    # Condition 1: B in range [60, 150]
    b_in_range = (b_channel >= 60) & (b_channel <= 150)
    
    # Condition 2: B is significantly higher than R and G
    # Use difference: B - max(R, G) should be large enough
    b_minus_rg = b_channel - np.maximum(r_channel, g_channel)
    b_dominant = b_minus_rg > 20  # B should be at least 20 higher than max(R,G)
    
    # Condition 3: R and G should be relatively low (to avoid white/gray)
    # R and G should be less than B, and not too high
    rg_low = (r_channel < b_channel) & (g_channel < b_channel) & (r_channel < 100) & (g_channel < 100)
    
    # Combine all conditions
    blue_mask = (b_in_range & b_dominant & rg_low).astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
    
    print(f"[AUTO] Found {num_labels-1} blue region(s) in image")  # -1 because label 0 is background
    
    if num_labels < 2:  # No objects found (label 0 is background)
        print(f"[AUTO] No blue objects detected. Try adjusting color range or use manual mode.")
        return None, None
    
    # Filter out components that touch image edges or are too large
    # Blue object should not be connected to image boundaries
    # Blue object should not exceed 1/6 of image area
    h, w = image_bgr.shape[:2]
    image_area = h * w
    max_area = image_area / 6.0  # Maximum allowed area (1/6 of image)
    
    valid_indices = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        # Get bounding box and area of this component
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Check if component touches any edge
        touches_left = (x <= 0)
        touches_right = (x + width >= w)
        touches_top = (y <= 0)
        touches_bottom = (y + height >= h)
        touches_edge = touches_left or touches_right or touches_top or touches_bottom
        
        # Check if component is too large
        too_large = area > max_area
        
        if touches_edge:
            edge_info = []
            if touches_left: edge_info.append("left")
            if touches_right: edge_info.append("right")
            if touches_top: edge_info.append("top")
            if touches_bottom: edge_info.append("bottom")
            print(f"[AUTO] Component {i} touches edge ({', '.join(edge_info)}), skipping")
        elif too_large:
            area_ratio = area / image_area
            print(f"[AUTO] Component {i} too large (area={area}, {area_ratio*100:.1f}% of image, max={max_area/image_area*100:.1f}%), skipping")
        else:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        print(f"[AUTO] No blue objects found that don't touch image edges.")
        return None, None
    
    # Find the largest component among valid (non-edge-touching) components
    # stats format: [x, y, width, height, area]
    valid_areas = [stats[i, cv2.CC_STAT_AREA] for i in valid_indices]
    largest_valid_idx = valid_indices[np.argmax(valid_areas)]
    largest_idx = largest_valid_idx
    
    # Get the mask for the largest component
    largest_mask = (labels == largest_idx).astype(np.uint8) * 255
    
    # Get bounding box of the largest component
    x = stats[largest_idx, cv2.CC_STAT_LEFT]
    y = stats[largest_idx, cv2.CC_STAT_TOP]
    w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    area = stats[largest_idx, cv2.CC_STAT_AREA]
    area_ratio = area / image_area
    
    print(f"[AUTO] Found blue object: area={area} pixels ({area_ratio*100:.1f}% of image), bbox=({x}, {y}, {w}, {h})")
    
    # Find five evenly distributed points within the blue region
    # Strategy: divide the region into a grid and select points from different areas
    blue_pixels = np.where(largest_mask > 0)
    if len(blue_pixels[0]) == 0:
        return None, None
    
    # Get all blue pixel coordinates
    blue_coords = np.column_stack((blue_pixels[1], blue_pixels[0]))  # (x, y) format
    
    # Get bounding box coordinates
    x_min, x_max = blue_coords[:, 0].min(), blue_coords[:, 0].max()
    y_min, y_max = blue_coords[:, 1].min(), blue_coords[:, 1].max()
    
    # Divide the region into a 3x2 grid (or similar) to get 5-6 regions
    # Then select one point from each of 5 regions
    num_points = 5
    
    points = []
    
    # Strategy: Divide into regions and pick center point of each region
    # Option 1: Divide by both x and y to create a grid
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Create a grid: roughly 2 columns x 3 rows (or 3x2) to get ~6 regions, pick 5
    # We'll use a simpler approach: divide into 5 regions based on position
    # Region 1: center
    # Region 2: top-left
    # Region 3: top-right
    # Region 4: bottom-left
    # Region 5: bottom-right
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Define 5 regions
    regions = [
        # Region 1: center
        ((center_x - x_range * 0.2, center_x + x_range * 0.2),
         (center_y - y_range * 0.2, center_y + y_range * 0.2)),
        # Region 2: top-left
        ((x_min, center_x),
         (y_min, center_y)),
        # Region 3: top-right
        ((center_x, x_max),
         (y_min, center_y)),
        # Region 4: bottom-left
        ((x_min, center_x),
         (center_y, y_max)),
        # Region 5: bottom-right
        ((center_x, x_max),
         (center_y, y_max)),
    ]
    
    for i, ((x_low, x_high), (y_low, y_high)) in enumerate(regions):
        # Find points in this region
        region_coords = blue_coords[
            (blue_coords[:, 0] >= x_low) & (blue_coords[:, 0] < x_high) &
            (blue_coords[:, 1] >= y_low) & (blue_coords[:, 1] < y_high)
        ]
        
        if len(region_coords) > 0:
            # Use the center point of this region
            region_center = np.mean(region_coords, axis=0)
            # Find the point closest to the region center
            closest = region_coords[np.argmin(np.sum((region_coords - region_center)**2, axis=1))]
            points.append([int(closest[0]), int(closest[1])])
    
    # If we don't have enough points, use a fallback: divide by y-coordinate
    if len(points) < num_points:
        # Divide by y-coordinate into 5 parts
        y_coords = blue_pixels[0]
        y_min, y_max = y_coords.min(), y_coords.max()
        y_step = (y_max - y_min) / num_points
        
        for i in range(num_points):
            y_target_low = y_min + i * y_step
            y_target_high = y_min + (i + 1) * y_step
            region_coords = blue_coords[
                (blue_coords[:, 1] >= y_target_low) & 
                (blue_coords[:, 1] < y_target_high)
            ]
            if len(region_coords) > 0:
                region_center = np.mean(region_coords, axis=0)
                closest = region_coords[np.argmin(np.sum((region_coords - region_center)**2, axis=1))]
                points.append([int(closest[0]), int(closest[1])])
    
    # Remove duplicates and ensure we have exactly 5 points
    unique_points = []
    seen = set()
    for p in points:
        p_tuple = tuple(p)
        if p_tuple not in seen:
            seen.add(p_tuple)
            unique_points.append(p)
            if len(unique_points) >= num_points:
                break
    
    if len(unique_points) < num_points:
        print(f"[AUTO] Warning: Only found {len(unique_points)} points, using what we have")
    
    print(f"[AUTO] Selected {len(unique_points)} points for SAM: {unique_points}")
    
    return unique_points, largest_mask


def overlay_mask(image_bgr, mask_bool, points, alpha=0.5):
    """Overlay mask and click points on image for visualization."""
    vis = image_bgr.copy()
    if mask_bool is not None:
        mask_uint8 = (mask_bool.astype(np.uint8) * 255)
        mask_color = np.zeros_like(image_bgr)
        mask_color[:, :, 0] = mask_uint8
        vis = cv2.addWeighted(image_bgr, 1.0, mask_color, alpha, 0)

    for (x, y) in points:
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
    return vis


def interactive_mask_for_camera(cam_name: str, image_bgr, predictor: SamPredictor, auto_detect=False):
    """
    Interactive SAM mask selection for a camera.
    
    Args:
        cam_name: Camera name
        image_bgr: Input image in BGR format
        predictor: SAM predictor
        auto_detect: If True, automatically detect blue object and use it
    """
    if predictor is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    click_points: list[list[int]] = []
    mask_result = None
    vis = image_bgr.copy()
    
    window_name = f"SAM Init - {cam_name}"
    
    # Auto-detect blue object if requested
    if auto_detect:
        print(f"[AUTO] Attempting to auto-detect blue object for {cam_name}...")
        detected_points, blue_mask = detect_largest_blue_object(image_bgr)
        if detected_points is not None and len(detected_points) > 0:
            # Use the detected points (should be 3 points evenly distributed)
            click_points.extend(detected_points)
            print(f"[AUTO] Auto-detected {len(detected_points)} points in blue region")
            # Automatically compute mask
            points = np.array(click_points)
            labels = np.ones(len(click_points), dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
            best_idx = int(np.argmax(scores))
            mask_result = masks[best_idx]
            print(f"[AUTO] Generated mask with score {scores[best_idx]:.3f}")
            # Show result and wait for user confirmation
            vis = overlay_mask(image_bgr, mask_result, click_points)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            print(f"[AUTO] Showing detected result. Press 's' to save, 'n' to skip, 'q' to quit")
            
            while True:
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(20) & 0xFF
                
                if key == ord('s'):
                    print(f"[AUTO] Saving mask for {cam_name}")
                    cv2.destroyWindow(window_name)
                    return mask_result
                elif key == ord('n'):
                    print(f"[AUTO] Skipping mask for {cam_name}")
                    cv2.destroyWindow(window_name)
                    return None
                elif key == ord('q'):
                    print(f"[AUTO] Quitting")
                    cv2.destroyWindow(window_name)
                    return None
        else:
            print(f"[AUTO] No blue object detected for {cam_name}, falling back to manual selection")
            print(f"[AUTO] You can still manually click points to create mask")

    def recompute_mask():
        nonlocal mask_result, vis
        if not click_points:
            mask_result = None
            vis = image_bgr.copy()
            return

        points = np.array(click_points)
        labels = np.ones(len(click_points), dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        mask_result = masks[best_idx]
        vis = overlay_mask(image_bgr, mask_result, click_points)
        print(f"[SAM] {cam_name}: {len(scores)} masks, best score {scores[best_idx]:.3f}")

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append([x, y])
            recompute_mask()
        elif event == cv2.EVENT_RBUTTONDOWN and click_points:
            click_points.pop()
            recompute_mask()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_cb)
    print(f"[SAM] {cam_name}: left click=FG, right click=undo, 'c'=clear, 's'=save, 'n'=skip")

    while True:
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('c'):
            click_points.clear()
            recompute_mask()
        elif key == ord('n'):
            print(f"[SAM] {cam_name}: skipped mask creation.")
            cv2.destroyWindow(window_name)
            return None
        elif key == ord('s'):
            if mask_result is None:
                print("[SAM] Need at least one foreground point before saving.")
                continue
            cv2.destroyWindow(window_name)
            return mask_result
        elif key == ord('q'):
            print(f"[SAM] {cam_name}: aborted via 'q'.")
            cv2.destroyWindow(window_name)
            return None


def run_initial_mask_workflow(camera_entries, predictor: SamPredictor, auto_detect=False):
    """Process masks for all camera entries."""
    if predictor is None or not camera_entries:
        return

    for entry in camera_entries:
        cam_name = entry["name"]
        image_bgr = entry["image"]
        mask_path = entry["mask_path"]
        mask = interactive_mask_for_camera(cam_name, image_bgr, predictor, auto_detect=auto_detect)
        if mask is None:
            continue
        mask_uint8 = (mask.astype(np.uint8) * 255)
        cv2.imwrite(str(mask_path), mask_uint8)
        print(f"[SAM] Saved {cam_name} mask to: {mask_path}")


def load_sam_predictor(checkpoint_path: str, model_type: str) -> SamPredictor:
    """Load SAM model and return predictor."""
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    print(f"[INFO] Loading SAM ({model_type}) from {checkpoint_path} onto {device}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)
    print("[INFO] SAM model ready.")
    return predictor


def gather_initial_entries(seq_root: Path):
    """Gather first frame images from both cameras."""
    entries = []
    cameras = [
        ("RealSense", seq_root / "realsense"),
        ("ZED", seq_root / "zed"),
    ]

    for cam_name, cam_root in cameras:
        rgb_path = cam_root / "rgb" / "000000.png"
        mask_path = cam_root / "masks" / "000000_mask.png"
        if not rgb_path.exists():
            print(f"[SAM] {cam_name}: first frame not found at {rgb_path}")
            continue
        image_bgr = cv2.imread(str(rgb_path))
        if image_bgr is None:
            print(f"[SAM] {cam_name}: failed to load {rgb_path}")
            continue
        
        # Ensure masks directory exists
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        
        entries.append({
            "name": cam_name,
            "image": image_bgr,
            "mask_path": mask_path,
        })

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Process SAM masks for recorded data (first frame of each camera)."
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
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use manual mode (disable auto-detection). Default: auto-detection is ON.",
    )
    args = parser.parse_args()

    seq_root = Path(args.data_path)
    if not seq_root.exists():
        print(f"[ERROR] Data path not found: {seq_root}")
        return

    # Find checkpoint
    sam_checkpoint_path = args.sam_checkpoint or find_checkpoint_file(args.checkpoints_root)
    if not sam_checkpoint_path:
        print("[ERROR] No SAM checkpoint found. Please specify --sam-checkpoint or ensure checkpoints/ contains a .pth file.")
        return

    # Load SAM model
    sam_predictor = load_sam_predictor(sam_checkpoint_path, args.sam_model_type)

    # Gather first frame images
    entries = gather_initial_entries(seq_root)
    if not entries:
        print("[ERROR] No initial frames found to annotate.")
        return

    # Process masks
    # Auto-detect is ON by default, unless --manual is specified
    auto_detect_enabled = not args.manual
    
    if auto_detect_enabled:
        print("[INFO] ========================================")
        print("[INFO] Auto-detection mode ENABLED (default)")
        print("[INFO] Will automatically find blue objects")
        print("[INFO] ========================================")
    else:
        print("[INFO] Manual mode: you will need to click points")
        print("[INFO] Auto-detection is disabled (--manual flag)")
    run_initial_mask_workflow(entries, sam_predictor, auto_detect=auto_detect_enabled)
    
    cv2.destroyAllWindows()
    print("[INFO] SAM processing complete.")


if __name__ == "__main__":
    main()

