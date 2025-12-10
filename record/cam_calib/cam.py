import pickle
import numpy as np
import yaml

from homography_utils import HomographyTransform


for cam_idx in [1, 2, 3, 4]:
    cam_cfg = yaml.safe_load(open(f"camera.yaml"))[f"cam{cam_idx}"]["camera_cfg"]

    hom = HomographyTransform(
        key=f"img{cam_idx}", transform_file="hom", cam_cfg=cam_cfg
    )

    intrinsics = np.array(cam_cfg["intrinsics"])

    extrinsics = hom.transform_matrix

    extrinsics[-1] += hom.pcs

    print(extrinsics.T)

    np.save(f"cam{cam_idx}_intrinsics.npy", intrinsics)
    np.save(f"cam{cam_idx}_extrinsics.npy", extrinsics.T)
