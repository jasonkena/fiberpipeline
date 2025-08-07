import os
import numpy as np
import warnings


def preprocess_vol(vol):
    assert vol.ndim == 4, f"Expected 4D image, got {vol.ndim}D"
    if vol.shape[1] != 4:
        warnings.warn(
            f"Expected 4 channels, got {vol.shape[1]} channels. "
            "This may lead to unexpected behavior."
        )
    # C, Z, Y, X
    vol = np.transpose(vol, (1, 0, 2, 3))

    return vol


def get_basename(path):
    return "-".join(os.path.basename(path).split("-")[:-1])
