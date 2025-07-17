import os
import numpy as np


def preprocess_vol(vol):
    assert vol.ndim == 4, f"Expected 4D image, got {vol.ndim}D"
    assert vol.shape[1] == 4, f"Expected 4 channels, got {vol.shape[1]}"
    # C, Z, Y, X
    vol = np.transpose(vol, (1, 0, 2, 3))

    return vol

def get_basename(path):
    return '-'.join(os.path.basename(path).split('-')[:-1])
