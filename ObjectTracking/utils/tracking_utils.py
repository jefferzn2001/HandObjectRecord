"""
Shared utilities for object tracking scripts.

Contains SAM3 loading/inference, mesh loading, FoundationPose estimator
construction, camera intrinsics helpers, and visualization.
"""

import logging
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBJECT_DIR = PROJECT_ROOT / "object"
FP_DIR = PROJECT_ROOT / "FoundationPose"
sys.path.insert(0, str(FP_DIR))

import nvdiffrast.torch as dr
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed


# ---------------------------------------------------------------------------
# SAM3 subprocess worker — keeps SAM3 in a separate process so its imports
# don't corrupt the main process's OpenGL / Qt5 context (which would crash
# cv2.namedWindow / cv2.imshow).
# ---------------------------------------------------------------------------

def _sam3_loop(input_q: mp.Queue, output_q: mp.Queue,
               confidence: float, project_root: str, fp_dir: str):
    """Entry point for the SAM3 background process."""
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    sys.path.insert(0, project_root)
    sys.path.insert(0, fp_dir)

    _shadow = os.path.join(project_root, "sam3")
    _removed = []
    for p in list(sys.path):
        resolved = str(Path(p).resolve()) if p else project_root
        if Path(resolved) == Path(project_root) or Path(resolved) == Path(_shadow):
            _removed.append(p)
            sys.path.remove(p)
    for key in [k for k in sys.modules if k == "sam3" or k.startswith("sam3.")]:
        del sys.modules[key]

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    for p in _removed:
        if p not in sys.path:
            sys.path.append(p)

    model = build_sam3_image_model(device="cuda", eval_mode=True,
                                    enable_segmentation=True)
    proc = Sam3Processor(model, confidence_threshold=confidence)
    output_q.put("ready")

    while True:
        item = input_q.get()
        if item is None:
            break
        frame_path, object_name = item
        color_rgb = np.load(frame_path)
        from PIL import Image
        pil_image = Image.fromarray(color_rgb)
        state = proc.set_image(pil_image)
        out = proc.set_text_prompt(state=state, prompt=object_name)
        masks, scores = out["masks"], out["scores"]
        if masks is not None and len(masks) > 0:
            best = torch.argmax(scores).item()
            mask_np = masks[best, 0].cpu().numpy().astype(np.uint8)
            torch.cuda.synchronize()
            if mask_np.sum() > 0:
                mask_out = os.path.splitext(frame_path)[0] + "_mask.npy"
                np.save(mask_out, mask_np)
                output_q.put(mask_out)
                continue
        output_q.put(None)


class SAM3Worker:
    """Persistent SAM3 subprocess that doesn't pollute the caller's GL state."""

    def __init__(self, confidence: float = 0.5):
        ctx = mp.get_context("spawn")
        self._in_q = ctx.Queue()
        self._out_q = ctx.Queue()
        self._tmpdir = tempfile.mkdtemp(prefix="sam3_ipc_")
        self._counter = 0
        self._pending = False
        self._pending_frame = ""
        self._proc = ctx.Process(
            target=_sam3_loop,
            args=(self._in_q, self._out_q, confidence,
                  str(PROJECT_ROOT), str(FP_DIR)),
            daemon=True,
        )
        self._proc.start()
        status = self._out_q.get(timeout=120)
        assert status == "ready", f"SAM3 worker failed to start: {status}"
        logging.info("SAM3 subprocess worker ready")

    def get_mask(self, color_rgb: np.ndarray, object_name: str) -> Optional[np.ndarray]:
        """Send a frame to the subprocess and get back a binary mask (blocking)."""
        self._counter += 1
        frame_path = os.path.join(self._tmpdir, f"frame_{self._counter}.npy")
        np.save(frame_path, color_rgb)
        self._in_q.put((frame_path, object_name))
        result = self._out_q.get(timeout=60)
        try:
            os.remove(frame_path)
        except OSError:
            pass
        if result is None:
            return None
        mask = np.load(result)
        try:
            os.remove(result)
        except OSError:
            pass
        return mask

    # -- Non-blocking API --------------------------------------------------

    def submit(self, color_rgb: np.ndarray, object_name: str) -> None:
        """Send a frame for detection without blocking. Call poll() to check."""
        if self._pending:
            return
        self._counter += 1
        frame_path = os.path.join(self._tmpdir, f"frame_{self._counter}.npy")
        np.save(frame_path, color_rgb)
        self._in_q.put((frame_path, object_name))
        self._pending = True
        self._pending_frame = frame_path

    def poll(self) -> tuple[bool, Optional[np.ndarray]]:
        """Check if a submitted result is ready. Returns (done, mask_or_None)."""
        if not self._pending:
            return False, None
        try:
            result = self._out_q.get_nowait()
        except Exception:
            return False, None
        self._pending = False
        try:
            os.remove(self._pending_frame)
        except OSError:
            pass
        if result is None:
            return True, None
        mask = np.load(result)
        try:
            os.remove(result)
        except OSError:
            pass
        return True, mask

    def shutdown(self):
        """Cleanly terminate the subprocess."""
        try:
            self._in_q.put(None)
            self._proc.join(timeout=10)
        except Exception:
            self._proc.kill()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __del__(self):
        self.shutdown()


def load_sam3(confidence: float = 0.5):
    """
    Load SAM3 model and processor for text-prompted segmentation.

    Args:
        confidence (float): Detection confidence threshold.

    Returns:
        tuple: (model, processor) ready for inference.
    """
    # Reason: the project root contains a sam3/ directory (the git repo) that
    # Python finds as a namespace package, shadowing the real editable-installed
    # sam3 package. Temporarily strip conflicting paths and clear any cached
    # namespace entry so the real package in site-packages is resolved.
    _sam3_shadow = str(PROJECT_ROOT / "sam3")
    _conflicting = []
    for p in list(sys.path):
        resolved = str(Path(p).resolve()) if p else str(PROJECT_ROOT)
        if Path(resolved) == PROJECT_ROOT or Path(resolved) == Path(_sam3_shadow):
            _conflicting.append(p)
            sys.path.remove(p)

    for key in [k for k in sys.modules if k == "sam3" or k.startswith("sam3.")]:
        del sys.modules[key]

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    for p in _conflicting:
        if p not in sys.path:
            sys.path.append(p)

    logging.info("Loading SAM3 model (this may take a moment on first run)...")
    model = build_sam3_image_model(
        device="cuda",
        eval_mode=True,
        enable_segmentation=True,
    )
    processor = Sam3Processor(model, confidence_threshold=confidence)
    logging.info("SAM3 model loaded")
    return model, processor


def get_sam3_mask(
    processor,
    color_rgb: np.ndarray,
    object_name: str,
) -> Optional[np.ndarray]:
    """
    Use SAM3 text-prompted segmentation to find and segment the object.

    Args:
        processor: Sam3Processor instance.
        color_rgb (np.ndarray): RGB image (H, W, 3), uint8.
        object_name (str): Text prompt for the object to detect.

    Returns:
        Optional[np.ndarray]: Binary mask (H, W) as uint8 (0 or 1),
                              or None if no detection.
    """
    from PIL import Image

    pil_image = Image.fromarray(color_rgb)
    inference_state = processor.set_image(pil_image)
    output = processor.set_text_prompt(state=inference_state, prompt=object_name)

    masks = output["masks"]
    scores = output["scores"]

    if masks is None or len(masks) == 0:
        return None

    # Reason: pick highest-confidence detection if multiple instances found
    best_idx = torch.argmax(scores).item()
    mask_np = masks[best_idx, 0].cpu().numpy().astype(np.uint8)

    if mask_np.sum() == 0:
        return None

    # Reason: CUDA kernels may still be in-flight; synchronize before
    # returning to caller so cv2.imshow doesn't race with the GPU.
    torch.cuda.synchronize()

    logging.info(
        f"SAM3 detected '{object_name}' with score {scores[best_idx]:.3f}, "
        f"mask pixels: {mask_np.sum()}"
    )
    return mask_np


def load_mesh(object_name: str) -> Tuple[str, Path]:
    """
    Locate the .obj mesh file for the given object.

    Args:
        object_name (str): Name matching a folder in object/.

    Returns:
        tuple: (mesh_path_str, mesh_dir) for the object.

    Raises:
        SystemExit: If object directory or mesh file not found.
    """
    mesh_dir = OBJECT_DIR / object_name
    if not mesh_dir.exists():
        logging.error(f"Object directory not found: {mesh_dir}")
        available = [
            d.name for d in OBJECT_DIR.iterdir() if d.is_dir()
        ]
        logging.info(f"Available objects: {available}")
        sys.exit(1)

    mesh_files = list(mesh_dir.glob("*.obj"))
    if not mesh_files:
        logging.error(f"No .obj file found in {mesh_dir}")
        sys.exit(1)

    mesh_path = mesh_files[0]
    logging.info(f"Using mesh: {mesh_path}")
    return str(mesh_path), mesh_dir


def build_estimator(
    mesh_path: str,
    debug_dir: str = "/tmp/fp_debug",
    est_refine_iter: int = 2,
    track_refine_iter: int = 2,
    debug: int = 0,
) -> Tuple[FoundationPose, trimesh.Trimesh, np.ndarray, np.ndarray]:
    """
    Build the FoundationPose estimator from a mesh file.

    Args:
        mesh_path (str): Path to the .obj mesh file.
        debug_dir (str): Directory for debug output.
        est_refine_iter (int): Refinement iterations for registration.
        track_refine_iter (int): Refinement iterations for tracking.
        debug (int): Debug level (0=off, 1=basic, 2=detailed).

    Returns:
        tuple: (estimator, mesh, to_origin, bbox).
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.vertices = mesh.vertices.astype(np.float32)
    mesh.vertex_normals = mesh.vertex_normals.astype(np.float32)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3).astype(np.float32)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    est.est_refine_iter = est_refine_iter
    est.track_refine_iter = track_refine_iter

    if debug < 1:
        logging.getLogger().setLevel(logging.WARNING)

    print("[build_estimator()] FoundationPose estimator ready", flush=True)
    return est, mesh, to_origin, bbox


def load_camera_serial(name: Optional[str] = None) -> Optional[str]:
    """
    Look up a RealSense serial number from camera_config.yaml.

    Args:
        name (Optional[str]): Camera name (e.g. ``"robotcam"``).
            If None, returns None immediately (use any available camera).

    Returns:
        Optional[str]: Serial number string, or None to use any camera.
    """
    if not name:
        return None

    config_path = PROJECT_ROOT / "camera_config.yaml"
    if not config_path.exists():
        logging.warning(
            f"camera_config.yaml not found — cannot look up '{name}'. "
            "Using any available camera."
        )
        return None

    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cameras = cfg.get("cameras", {})
    if name not in cameras:
        available = list(cameras.keys())
        logging.error(f"Camera '{name}' not in config. Available: {available}")
        return None

    serial = cameras[name]["serial"]
    logging.info(f"Using camera '{name}' (serial {serial})")
    return serial


def intrinsics_to_K(intr) -> np.ndarray:
    """
    Convert RealSense intrinsics to a 3x3 camera matrix.

    Args:
        intr: pyrealsense2 intrinsics object.

    Returns:
        np.ndarray: 3x3 camera intrinsic matrix.
    """
    return np.array(
        [
            [float(intr.fx), 0.0, float(intr.ppx)],
            [0.0, float(intr.fy), float(intr.ppy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def print_pose(pose: np.ndarray, object_name: str) -> None:
    """
    Print pose to console.

    Args:
        pose (np.ndarray): 4x4 pose matrix.
        object_name (str): Name of the tracked object.
    """
    from scipy.spatial.transform import Rotation as R

    t = pose[:3, 3]
    quat = R.from_matrix(pose[:3, :3]).as_quat()
    logging.info(
        f"[{object_name}] pos=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) "
        f"quat=({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})"
    )


def draw_tracking_vis(
    color_bgr: np.ndarray,
    pose: Optional[np.ndarray],
    to_origin: np.ndarray,
    bbox: np.ndarray,
    K: np.ndarray,
    initialized: bool,
    object_name: str,
) -> np.ndarray:
    """
    Render the tracking overlay (3D bounding box + axes) on a BGR image.

    Args:
        color_bgr (np.ndarray): BGR camera frame.
        pose (Optional[np.ndarray]): Current 4x4 pose, or None.
        to_origin (np.ndarray): Mesh-to-origin transform.
        bbox (np.ndarray): Bounding box corners (2, 3).
        K (np.ndarray): Camera intrinsics (3, 3).
        initialized (bool): Whether tracking is active.
        object_name (str): Object name for HUD.

    Returns:
        np.ndarray: BGR image with overlay drawn.
    """
    vis_bgr = color_bgr.copy()
    if initialized and pose is not None:
        center_pose = pose @ np.linalg.inv(to_origin)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        vis_rgb = draw_posed_3d_box(K, img=vis_rgb, ob_in_cam=center_pose, bbox=bbox)
        vis_rgb = draw_xyz_axis(
            vis_rgb,
            ob_in_cam=center_pose,
            scale=0.1,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    return vis_bgr


