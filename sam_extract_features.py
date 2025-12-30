#!/usr/bin/env python3
"""
Step 1: Extract and save SAM image embeddings (slow step, run once).

This script loads images, runs SAM to extract image embeddings,
and saves them for fast reuse in the next step.

Usage:
    python sam_extract_features.py --data_path data/test_wei_01
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


def extract_and_save_features(data_path: Path, predictor: SamPredictor):
    """Extract SAM features for all camera images and save them."""
    cameras = [
        ("realsense", data_path / "realsense"),
        ("zed", data_path / "zed"),
    ]
    
    for cam_name, cam_root in cameras:
        rgb_path = cam_root / "rgb" / "000000.png"
        if not rgb_path.exists():
            print(f"[SKIP] {cam_name}: first frame not found at {rgb_path}")
            continue
        
        print(f"\n[PROCESSING] {cam_name}...")
        
        # Load image
        image_bgr = cv2.imread(str(rgb_path))
        if image_bgr is None:
            print(f"[ERROR] {cam_name}: failed to load {rgb_path}")
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Extract features (this is the slow step)
        print(f"[EXTRACT] Running SAM encoder for {cam_name}...")
        predictor.set_image(image_rgb)
        
        # Get image embedding
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        
        # Save embedding and metadata
        features_dir = cam_root / "sam_features"
        features_dir.mkdir(exist_ok=True)
        
        embedding_path = features_dir / "000000_embedding.npy"
        metadata_path = features_dir / "000000_metadata.npy"
        
        # Save embedding
        np.save(embedding_path, image_embedding)
        print(f"[SAVE] Saved embedding to {embedding_path}")
        
        # Save metadata (image size, etc.)
        metadata = {
            'original_size': predictor.original_size,  # (H, W)
            'input_size': predictor.input_size,  # (H, W)
            'image_shape': image_bgr.shape,  # (H, W, 3)
        }
        np.save(metadata_path, metadata)
        print(f"[SAVE] Saved metadata to {metadata_path}")
        
        # Also save a copy of the image for reference
        image_save_path = features_dir / "000000_image.png"
        cv2.imwrite(str(image_save_path), image_bgr)
        print(f"[SAVE] Saved image copy to {image_save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save SAM image embeddings (Step 1 - slow, run once)."
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
        print("[ERROR] No SAM checkpoint found. Please specify --sam-checkpoint or ensure checkpoints/ contains a .pth file.")
        return

    # Load SAM model
    predictor = load_sam_predictor(sam_checkpoint_path, args.sam_model_type)

    # Extract and save features
    print("\n" + "="*60)
    print("STEP 1: Extracting SAM features (this may take a while...)")
    print("="*60)
    extract_and_save_features(data_path, predictor)
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETE: Features saved!")
    print("="*60)
    print(f"\nNext step: Run sam_detect_blue.py to quickly detect blue objects:")
    print(f"  python sam_detect_blue.py --data_path {data_path}")


if __name__ == "__main__":
    main()

