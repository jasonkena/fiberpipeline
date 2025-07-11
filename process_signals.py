import os
import glob
import numpy as np

from scipy.interpolate import interpn
from skopt.space import Integer
from scipy.stats import pearsonr
from multiprocessing import Pool

from nicety.conf import get_conf

"""

Pipeline:
    * remove tails on each Skeleton and interpolate to original length
    * compute the geodesic midpoint and search for signal midpoint in a small interval (plus-minus 5%) around the geodesic midpoint
        - calculated as the point that, when seperating the two length-normalized "halves", maximizes the Pearson statistics between each half
    * calculate I_baseline as the mean of the
    * 

"""

def interp2d(x, size):
    return interpn(
        points=np.expand_dims(np.arange(len(x)), 0),
        values=x,
        xi=np.expand_dims(np.linspace(0, len(x)-1, size), -1),
    )

def remove_tails(signal):
    return interp2d(np.delete(signal, signal==0), conf.skeletonize_fibers.num_centerline_points)

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
    halves = list(map(lambda x: interp2d(x, int(len(signal)/2)), halves))
    signal = np.concatenate([halves[0], halves[1]])
    I_baseline = signal[int(signal.shape[0]*0.45):int(signal.shape[0]*0.55)].mean()
    if I_baseline == 0:
        I_baseline = 1
    return (signal - I_baseline) / I_baseline

def normalize_signals(signals):
    # signal data is stored in channel 2
    signal_midpoint = get_midpoint(signals[2])
    return list(map(lambda x: normalize_signal(x, signal_midpoint), signals))

def geodesic(skel, center=None):
    dist = np.cumsum(np.linalg.norm(np.diff(skel.vertices, axis=0), axis=1))
    dist = np.concatenate([[0], dist])
    if center != None:
        for i in range(len(dist)):
            if i <= center:
                dist[i] = dist[i] * i/len(dist)
            else:
                dist[i] = dist[center] + (dist[-1] - dist[center]) * (i - center) / (len(dist)-1-center)
        dist -= dist[center]
    return dist

def normalize_all_signals(conf):
    files = sorted(glob.glob(os.path.join(conf.output_path, "*-signals.npz")))
    for file in files:
        print(f"Processing {file}")
        signals = np.load(file, allow_pickle=True)
        base_name = os.path.basename(file).replace("-signals.npz", "")
        signals, signal_labels = signals["signals"], signals["signal_labels"]
        skels = np.load(
            file.replace("-signals.npz", "-fiber_skel.npz"), allow_pickle=True
        )["skels"].tolist()
        
        for channel_idx in range(signals.shape[0]):
            signals[channel_idx] = list(map(remove_tails, signals[channel_idx]))
        midpoints = list(map(get_midpoint, signals[2]))
        signals = list(np.moveaxis(np.array(signals), 1, 0))
        with Pool(conf.skeletonize_fibers.num_cpus) as p:
            signals = p.map(normalize_signals, signals)
        signals = np.moveaxis(signals, 1, 0)
        geodesics = [geodesic(skel, center=midpoint) for skel, midpoint in zip(skels, midpoints)]
        
        np.savez(
            os.path.join(
                conf.output_path,
                base_name + "-processed_signals.npz",
            ),
            signals=signals,
            signal_labels=signal_labels,
            geodesics=geodesics
        )

if __name__ == "__main__":
    conf = get_conf()
    normalize_all_signals(conf)
