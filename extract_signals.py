import os
import h5py
import glob
import numpy as np

from tqdm import tqdm
from nicety.conf import get_conf
from scipy.interpolate import interpn
import imageio
from utils import preprocess_vol

from scipy.interpolate import interpn

def extract_signals(vol, skels, anisotropy, method: str):
    assert vol.ndim == 3, "Volume must be 3D"
    skel_lens = np.array([skel.vertices.shape[0] for skel in skels])
    # already in isotropic space
    all_skel_vertices = np.concatenate([skel.vertices for skel in skels], axis=0)

    points = [np.arange(vol.shape[i], dtype=float) * anisotropy[i] for i in range(3)]
    signals = interpn(
        points,
        vol,
        all_skel_vertices,
        method=method,
        bounds_error=False,
        fill_value=0,
    )
    assert np.all(signals >= 0), "Signals must be non-negative"
    # [ [signal for vertex in skel] for skel in skels ]
    signals = np.split(signals, np.cumsum(skel_lens)[:-1])
    return signals


def generate_signals(conf):
    files = sorted(glob.glob(os.path.join(conf.output_path, "*fiber_skel.npz")))
    for file in tqdm(files, desc="Processing files"):
        skels = np.load(file, allow_pickle=True)["skels"].tolist()
        base_name = os.path.basename(file).replace("-fiber_skel.npz", "")
        signal_labels = []
        signals = []

        im_vol = preprocess_vol(
            imageio.volread(os.path.join(conf.dataset_path, base_name + ".tif"))
        )
        assert im_vol.ndim == 4
        for i in range(im_vol.shape[0]):
            signal_labels.append(f"im_{i}")
            signals.append(
                extract_signals(
                    im_vol[i],
                    skels,
                    conf.anisotropy,
                    method="linear",
                )
            )
        del im_vol
        fiber_seg = h5py.File(
            os.path.join(conf.output_path, base_name + "-masked_fiber_seg.h5"),
        )
        assert (
            len(fiber_seg.keys()) == 1
        ), "Expected only one key in the fiber segmentation file"
        fiber_seg = fiber_seg[list(fiber_seg.keys())[0]][:]
        signals.append(
            extract_signals(
                fiber_seg,
                skels,
                conf.anisotropy,
                method="nearest",
            )
        )
        signal_labels.append("fiber_seg")
        del fiber_seg

        cell_seg = h5py.File(
            os.path.join(conf.output_path, base_name + "-cell_seg.h5"),
        )
        assert (
            len(cell_seg.keys()) == 1
        ), "Expected only one key in the cell segmentation file"
        cell_seg = cell_seg[list(cell_seg.keys())[0]][:]
        signals.append(
            extract_signals(
                cell_seg,
                skels,
                conf.anisotropy,
                method="nearest",
            )
        )
        signal_labels.append("cell_seg")
        del cell_seg
        
        np.savez(
            os.path.join(
                conf.output_path,
                base_name + "-signals.npz",
            ),
            signals=signals,
            signal_labels=signal_labels,
        )


if __name__ == "__main__":
    conf = get_conf()
    generate_signals(conf)
