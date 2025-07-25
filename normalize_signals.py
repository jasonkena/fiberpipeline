import os
from tqdm import tqdm
import glob
import numpy as np

from scipy.interpolate import interpn, interp1d
from skopt.space import Integer
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool

from utils import get_basename
from nicety.conf import get_conf
from tslearn.metrics import dtw, soft_dtw, dtw_path
from tslearn.barycenters import softdtw_barycenter

from skeletonize_fibers import points_to_skeleton
from joblib import Parallel, delayed

from collections import defaultdict


def interp2d(x, size):
    return interpn(
        points=np.expand_dims(np.arange(len(x)), 0),
        values=x,
        xi=np.expand_dims(np.linspace(0, len(x) - 1, size), -1),
    )


def dtw_distance(data, index):
    x, y = data[:index], data[index:]
    return soft_dtw(x, y)


def fix_dtw_path(path):
    # consists of pairs [(i, j)], eg: [(0, 0), (1, 1), (1, 2), (2, 3)], forcing them to be a function (every x only has one y)
    data = defaultdict(list)
    for i, j in path:
        data[i].append(j)
    data = {k: np.mean(v) for k, v in data.items()}
    data = sorted(data.items())
    data = np.array(data)
    return data


def recenter(signals, reference_key, skel, center, num_centerline_points):
    assert num_centerline_points % 2 == 0, "num_centerline_points must be even"
    signals = signals.copy()
    assert all([x not in signals for x in ["skel_z", "skel_y", "skel_x"]])
    signals["skel_z"] = skel[:, 0]
    signals["skel_y"] = skel[:, 1]
    signals["skel_x"] = skel[:, 2]

    left_signals = {key: signals[key][:center][::-1] for key in signals}
    right_signals = {key: signals[key][center:] for key in signals}

    barycenter = softdtw_barycenter(
        [
            left_signals[reference_key],
            right_signals[reference_key],
        ]
    )

    left_to_barycenter = fix_dtw_path(
        dtw_path(left_signals[reference_key], barycenter)[0]
    )
    assert np.all(
        left_to_barycenter[:, 0] == np.arange(left_signals[reference_key].shape[0])
    )
    # [0, 1]
    left_to_barycenter = left_to_barycenter[:, 1] / barycenter.shape[0]

    right_to_barycenter = fix_dtw_path(
        dtw_path(right_signals[reference_key], barycenter)[0]
    )
    assert np.all(
        right_to_barycenter[:, 0] == np.arange(right_signals[reference_key].shape[0])
    )
    # [0, 1]
    right_to_barycenter = right_to_barycenter[:, 1] / barycenter.shape[0]

    for key in signals:
        is_float = np.issubdtype(signals[key].dtype, np.floating)
        interp_left = interp1d(
            left_to_barycenter,
            left_signals[key],
            kind="linear" if is_float else "nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interp_right = interp1d(
            right_to_barycenter,
            right_signals[key],
            kind="linear" if is_float else "nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        left_signals[key] = interp_left(np.linspace(0, 1, num_centerline_points // 2))
        right_signals[key] = interp_right(np.linspace(0, 1, num_centerline_points // 2))

    final_signals = {}
    for key in signals:
        final_signals[key] = np.concatenate(
            [left_signals[key][::-1], right_signals[key]]
        )
    skel = np.stack(
        [
            final_signals.pop("skel_z"),
            final_signals.pop("skel_y"),
            final_signals.pop("skel_x"),
        ],
        axis=-1,
    )

    return final_signals, skel


def get_tail_crop_points(signal, extrapolate, gaussian_filter_ratio):
    # find lowest point for signal
    smoothed_signal = gaussian_filter1d(
        signal, int(gaussian_filter_ratio * len(signal))
    )
    assert extrapolate[0] < 0 and extrapolate[1] > 1
    num_points = signal.shape[0]
    start = int((0 - extrapolate[0]) / (extrapolate[1] - extrapolate[0]) * num_points)
    stop = int((1 - extrapolate[0]) / (extrapolate[1] - extrapolate[0]) * num_points)

    while (
        start > 0
        and signal[start] > 0
        and smoothed_signal[start - 1] < smoothed_signal[start]
    ):
        start -= 1
    while (
        stop < len(signal) - 1
        and signal[stop] > 0
        and smoothed_signal[stop + 1] < smoothed_signal[stop]
    ):
        stop += 1

    return start, stop, smoothed_signal


def get_midpoint(signal, midpoint_range):
    return min(
        range(
            int(len(signal) * midpoint_range[0]), int(len(signal) * midpoint_range[1])
        ),
        key=lambda x: dtw_distance(signal, x),
    )


def normalize_signal(signal, midpoint):
    halves = signal[:midpoint], signal[midpoint:]
    halves = list(map(lambda x: interp2d(x, int(len(signal) / 2)), halves))
    signal = np.concatenate([halves[0], halves[1]])
    I_baseline = signal[
        int(signal.shape[0] * 0.45) : int(signal.shape[0] * 0.55)
    ].mean()
    if I_baseline == 0:
        I_baseline = 1
    return (signal - I_baseline) / I_baseline


def normalize_signals(signals):
    # signal data is stored in channel 2
    signal_midpoint = get_midpoint(signals[2])
    return list(map(lambda x: normalize_signal(x, signal_midpoint), signals))


def geodesic(skel, center):
    dist = np.cumsum(np.linalg.norm(np.diff(skel.vertices, axis=0), axis=1))
    dist = np.concatenate([[0], dist])
    total_dist = dist[-1]
    dist_center = dist[center]

    for idx in range(len(dist)):
        if idx <= center:
            dist[idx] = (
                total_dist * (idx / center) * ((0.5 * len(skel.vertices)) / center)
            )
        else:
            dist[idx] = (
                total_dist
                * ((idx - center) / (len(skel.vertices) - center))
                * ((0.5 * len(skel.vertices)) / (len(skel.vertices) - center))
                + dist[center]
            )

    dist -= dist[center]
    return dist


def _normalize_all_signals(
    skel, signals, extrapolate, gaussian_filter_ratio, midpoint_range
):
    # calculate crop based on fiber channel
    original_im_1 = signals["im_1"]
    start, stop, smoothed = get_tail_crop_points(
        signals["im_1"], extrapolate, gaussian_filter_ratio
    )

    skel = skel[start : stop + 1]
    signals = {k: v[start : stop + 1] for k, v in signals.items()}

    midpoint = get_midpoint(signals["im_2"], midpoint_range)
    signals, skel = recenter(
        signals, "im_2", skel, midpoint, conf.skeletonize_fibers.num_centerline_points
    )

    import magicpickle as mp

    mp.send((signals["im_1"], original_im_1, smoothed, signals["im_2"]))
    print(midpoint)
    breakpoint()

    import magicpickle as mp

    im_1, original_im_1, smoothed, im_2 = mp.receive()

    import matplotlib.pyplot as plt

    plt.plot(im_1)
    plt.plot(original_im_1)
    plt.plot(smoothed)
    plt.plot(im_2)
    plt.show()

    breakpoint()


def normalize_all_signals(conf):
    files = sorted(glob.glob(os.path.join(conf.output_path, "*-signals.npz")))
    files = [x for x in files if "0702-2-C4-DG-40X002" in x]
    for file in (pbar := tqdm(files)):
        pbar.set_description(get_basename(file))
        signals = np.load(file, allow_pickle=True)
        base_name = os.path.basename(file).replace("-signals.npz", "")
        signals = signals["signals"].item()
        # convert to list of vertices
        skels = np.load(
            file.replace("-signals.npz", "-fiber_skel.npz"), allow_pickle=True
        )["skels"].tolist()
        skels = [skel.vertices for skel in skels]
        assert all([len(signals[x]) == len(skels) for x in signals])

        # res = list(
        #     tqdm(
        #     Parallel(n_jobs=conf.num_cpus, return_as="generator")(
        #         delayed(_normalize_all_signals)(
        #                 skels[i], {name: signals[name][i] for name in signals},
        #                 extrapolate=conf.skeletonize_fibers.extrapolate
        #             )
        #         for i in range(len(skels))
        #     ),
        #     total=len(skels),
        #     leave=False,
        #     )
        # )
        _normalize_all_signals(
            skels[1],
            {name: signals[name][1] for name in signals},
            extrapolate=conf.skeletonize_fibers.extrapolate,
            gaussian_filter_ratio=conf.normalize_signals.gaussian_filter_ratio,
            midpoint_range=conf.normalize_signals.midpoint_range,
        )
        breakpoint()

        for channel_idx in range(5):
            signals[channel_idx] = list(map(remove_tails, signals[channel_idx]))
        with Pool(conf.num_cpus) as p:
            midpoints = list(p.map(get_midpoint, signals[2]))
            signals = list(np.moveaxis(np.array(signals), 1, 0))
            signals = p.map(normalize_signals, signals)
            signals = np.moveaxis(signals, 1, 0)
            geodesics = [
                geodesic(skel, center=midpoint)
                for skel, midpoint in zip(skels, midpoints)
            ]

        np.savez(
            os.path.join(
                conf.output_path,
                base_name + "-normalized_signals.npz",
            ),
            signals=signals,
            signal_labels=signal_labels,
            geodesics=geodesics,
        )


if __name__ == "__main__":
    conf = get_conf()
    normalize_all_signals(conf)
