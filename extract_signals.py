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
from skopt import gp_minimize
from skopt.space import Integer
from scipy.stats import pearsonr
from multiprocessing import Pool

"""
Following Yixiao's method:
    * interpolate to a predefined length
    * calculate the signal midpoint
        - take a window of 20% of the total geodesic distance around the geodesic midpoint
        - select the point that maximizes the Pearson correlation between the right and left halves
    * interpolate each half to the same length as above and find the average recorded intensity (I.baseline)
    * normalize using I.baseline ((I - I.baseline) / I.baseline)
"""

def interp2d(x, size):
    return interpn(
        points=np.expand_dims(np.arange(len(x)), 0), 
        values=x,
        xi=np.expand_dims(np.linspace(0, len(x)-1, size), -1),
    )

def remove_tails(signal):
    return interp2d(np.delete(signal, signal==-1), conf.skeletonize_fibers.num_centerline_points)

def uneven_pearsonr(data, index):
    x, y = data[:index], data[index:]
    x = interp2d(x, len(data))
    y = interp2d(y, len(data))
    return pearsonr(x, y).statistic

def get_midpoint(signal):
    geodesic_low, geodesic_high = int(len(signal)*0.45), int(len(signal)*0.55)
    bounds = np.arange(geodesic_low, geodesic_high)
    pearson_scores = list(map(lambda x: uneven_pearsonr(signal, x), bounds))
    signal_midpoint = np.argmax(pearson_scores) + geodesic_low
    return signal_midpoint

def normalize_signal(signal, midpoint):
    halves = signal[:midpoint], signal[midpoint:]
    halves = list(map(lambda x: interp2d(x, len(signal)), halves))
    I_baseline = np.concatenate([halves[0], halves[1]]).mean()
    if I_baseline == 0:
        I_baseline = 1
    return (signal - I_baseline) / I_baseline

def normalize_signals(signals):
    # signal data is stored in channel 2
    signal_midpoint = get_midpoint(signals[2])
    return list(map(lambda x: normalize_signal(x, signal_midpoint), signals))

def extract_signals(vol, skels, anisotropy):
    assert vol.ndim == 3, "Volume must be 3D"
    skel_lens = np.array([skel.vertices.shape[0] for skel in skels])
    # already in isotropic space
    all_skel_vertices = np.concatenate([skel.vertices for skel in skels], axis=0)

    points = [np.arange(vol.shape[i], dtype=float) * anisotropy[i] for i in range(3)]
    signals = interpn(
        points,
        vol,
        all_skel_vertices,
        method="linear",
        bounds_error=False,
        fill_value=-1,
    )
    # [ [signal for vertex in skel] for skel in skels ]
    signals = np.split(signals, np.cumsum(skel_lens)[:-1])
    return signals


def generate_signals(conf):
    files = sorted(glob.glob(os.path.join(conf.output_path, "*fiber_skel.npz")))
    for file in tqdm(files, desc="Processing files"):
        skels = np.load(file, allow_pickle=True)["skels"].tolist()
        base_name = os.path.basename(file).replace("_fiber_skel.npz", "")
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
                )
            )
        del im_vol
        fiber_seg = h5py.File(
            os.path.join(conf.output_path, base_name + "_fiber_seg.h5"),
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
            )
        )
        signal_labels.append("fiber_seg")
        del fiber_seg

        cell_seg = h5py.File(
            os.path.join(conf.output_path, base_name + "_cell_seg.h5"),
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
            )
        )
        signal_labels.append("cell_seg")
        del cell_seg

        for channel_idx in range(4):
            signals[channel_idx] = list(map(remove_tails, signals[channel_idx]))
        
        # this is kinda wasteful
        im2 = signals[2]
        midpoints = list(map(get_midpoint, im2)) 

        signals = list(np.moveaxis(np.array(signals), 1, 0))
        with Pool(conf.skeletonize_fibers.num_cpus) as p:
            signals = p.map(normalize_signals, signals)
        signals = np.moveaxis(signals, 1, 0)

        np.savez(
            os.path.join(
                conf.output_path,
                base_name + "_signals.npz",
            ),
            signals=signals,
            signal_labels=signal_labels,
            midpoints=midpoints
        )


if __name__ == "__main__":
    conf = get_conf()
    generate_signals(conf)
