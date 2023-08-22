import numpy as np


class Camera:
    def __init__(self, intrinsics, extrinsics):
        assert type(intrinsics) is np.ndarray and intrinsics.shape == (3, 3), "intrinsics should be 3x3 ndarray"
        assert type(extrinsics) is np.ndarray and extrinsics.shape == (3, 4), "extrinsics should be 3x4 ndarray"
        
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics