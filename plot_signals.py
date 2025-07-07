import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interpn
from skopt.space import Integer
from scipy.stats import pearsonr
from multiprocessing import Pool

from nicety.conf import get_conf

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

def geodesic(skel, center: bool, offset=None):
    dist = np.cumsum(np.linalg.norm(np.diff(skel.vertices, axis=0), axis=1))
    dist = np.concatenate([[0], dist])
    if center:
        dist -= dist.mean()
    if offset != None:
        dist -= dist[offset]
    return dist

def plot_signals(conf):
    files = sorted(glob.glob(os.path.join(conf.output_path, "*_signals.npz")))
    for file in files:
        print(f"Processing {file}")
        signals = np.load(file, allow_pickle=True)
        signals, signal_labels, = signals["signals"], signals["signal_labels"]
        skels = np.load(
            file.replace("_signals.npz", "_fiber_skel.npz"), allow_pickle=True
        )["skels"].tolist()

        for channel_idx in range(4):
            signals[channel_idx] = list(map(remove_tails, signals[channel_idx]))
        midpoints = list(map(get_midpoint, signals[2]))
        signals = list(np.moveaxis(np.array(signals), 1, 0))
        with Pool(conf.skeletonize_fibers.num_cpus) as p:
            signals = p.map(normalize_signals, signals)
        signals = np.moveaxis(signals, 1, 0)
        geodesics = [geodesic(skel, center=True, offset=midpoint) for skel, midpoint in zip(skels, midpoints)]

        for i, signal_label in enumerate(signal_labels):
            if not "im" in signal_label:
                print(f"Skipping {signal_label}")
                continue

            # Plot each skeleton's geodesic vs signal (no individual labels)
            for j, (geodesic_vals, signal_vals) in enumerate(
                zip(geodesics, signals[i])
            ):
                plt.plot(geodesic_vals, signal_vals, alpha=0.3, linewidth=0.5)

            plt.xlabel("Geodesic Distance")
            plt.ylabel("Signal Value")
            plt.title(
                f"Geodesics vs Signal - {signal_label} (n={len(geodesics)} skeletons) for {os.path.basename(file)}"
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    conf = get_conf()
    plot_signals(conf)
