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

from skeletonize_fibers import points_to_skeleton
from joblib import Parallel, delayed

from collections import defaultdict


def uneven_pearsonr(data, index):
    x, y = data[:index][::-1], data[index:]

    # interpolate using last value
    max_length = max(len(x), len(y))
    new_x = np.ones(max_length) * x[-1]
    new_x[: len(x)] = x
    new_y = np.ones(max_length) * y[-1]
    new_y[: len(y)] = y

    return pearsonr(new_x, new_y).statistic


def recenter(signals, skel, center, num_centerline_points):
    assert num_centerline_points % 2 == 0, "num_centerline_points must be even"
    signals = signals.copy()
    assert all([x not in signals for x in ["skel_z", "skel_y", "skel_x"]])
    signals["skel_z"] = skel[:, 0]
    signals["skel_y"] = skel[:, 1]
    signals["skel_x"] = skel[:, 2]

    # ::-1 technically not necessary
    left_signals = {key: signals[key][:center][::-1] for key in signals}
    right_signals = {key: signals[key][center:] for key in signals}

    for key in signals:
        is_float = np.issubdtype(signals[key].dtype, np.floating)
        interp_left = interp1d(
            np.linspace(0, 1, left_signals[key].shape[0]),
            left_signals[key],
            kind="linear" if is_float else "nearest",
        )
        interp_right = interp1d(
            np.linspace(0, 1, right_signals[key].shape[0]),
            right_signals[key],
            kind="linear" if is_float else "nearest",
        )
        left_signals[key] = interp_left(np.linspace(0, 1, num_centerline_points // 2))
        right_signals[key] = interp_right(np.linspace(0, 1, num_centerline_points // 2))

    final_signals = {}
    for key in signals:
        final_signals[key] = np.concatenate(
            [left_signals[key][::-1], right_signals[key]]
        ).astype(signals[key].dtype)
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
        key=lambda x: uneven_pearsonr(signal, x),
    )


# def normalize_signal(signal, midpoint):
#     I_baseline = signal[
#         int(signal.shape[0] * 0.45) : int(signal.shape[0] * 0.55)
#     ].mean()
#     if I_baseline == 0:
#         I_baseline = 1
#     return (signal - I_baseline) / I_baseline


def get_geodesic(skel, center):
    dist = np.cumsum(np.linalg.norm(np.diff(skel, axis=0), axis=1))
    dist -= dist[center]

    return dist


def _normalize_all_signals(
    skel, signals, extrapolate, gaussian_filter_ratio, midpoint_range
):
    # calculate crop based on fiber channel
    start, stop, smoothed = get_tail_crop_points(
        signals["im_1"], extrapolate, gaussian_filter_ratio
    )

    skel = skel[start : stop + 1]
    signals = {k: v[start : stop + 1] for k, v in signals.items()}

    # midpoint based on im_2
    midpoint = get_midpoint(signals["im_2"], midpoint_range)
    # now 500 on each side
    signals, skel = recenter(
        signals, skel, midpoint, conf.skeletonize_fibers.num_centerline_points
    )
    geodesic = get_geodesic(skel, midpoint)
    signals["geodesic"] = geodesic

    return signals, skel


def normalize_all_signals(conf):
    files = sorted(glob.glob(os.path.join(conf.output_path, "*-signals.npz")))
    for file in (pbar := tqdm(files)):
        pbar.set_description(get_basename(file))
        signals = np.load(file, allow_pickle=True)
        base_name = os.path.basename(file).replace("-signals.npz", "")
        signals = signals["signals"].item()
        # convert to list of vertices
        skels = np.load(
            file.replace("-signals.npz", "-fiber_skel.npz"), allow_pickle=True
        )["skels"].tolist()
        skel_ids = np.array([skel.id for skel in skels])
        skels = [skel.vertices for skel in skels]
        assert all([len(signals[x]) == len(skels) for x in signals])

        res = list(
            tqdm(
                Parallel(n_jobs=conf.num_cpus, return_as="generator")(
                    delayed(_normalize_all_signals)(
                        skels[i],
                        {name: signals[name][i] for name in signals},
                        extrapolate=conf.skeletonize_fibers.extrapolate,
                        gaussian_filter_ratio=conf.normalize_signals.gaussian_filter_ratio,
                        midpoint_range=conf.normalize_signals.midpoint_range,
                    )
                    for i in range(len(skels))
                ),
                total=len(skels),
                leave=False,
            )
        )
        signals = defaultdict(list)
        skels = []
        for sig, skel in res:
            for key in sig:
                signals[key].append(sig[key])
            skels.append(skel)

        np.savez(
            os.path.join(
                conf.output_path,
                base_name + "-normalized_signals.npz",
            ),
            signals=signals,
            skels=skels,
            skel_ids=skel_ids,
        )


if __name__ == "__main__":
    conf = get_conf()
    normalize_all_signals(conf)
