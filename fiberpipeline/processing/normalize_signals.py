import os
from tqdm import tqdm
import glob
import numpy as np

from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

from fiberpipeline.utils import get_basename
from nicety.conf import get_conf

from joblib import Parallel, delayed
import optuna
import warnings

from collections import defaultdict


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


def apply_midpoint_scale(
    data, index, scale, num_centerline_points, calculate_pearson=False
):
    dtype = data.dtype
    is_float = np.issubdtype(data.dtype, np.floating)
    index = int(data.shape[0] * index)
    x, y = data[:index][::-1], data[index:]

    if scale < 0:
        x = x[: int((1 - abs(scale)) * x.shape[0])]
    elif scale > 0:
        y = y[: int((1 - abs(scale)) * y.shape[0])]

    x = interp1d(
        np.linspace(0, 1, x.shape[0]),
        x,
        kind="linear" if is_float else "nearest",
    )(np.linspace(0, 1, num_centerline_points // 2)).astype(dtype)
    y = interp1d(
        np.linspace(0, 1, y.shape[0]),
        y,
        kind="linear" if is_float else "nearest",
    )(np.linspace(0, 1, num_centerline_points // 2)).astype(dtype)

    data = np.concatenate([x[::-1], y])
    if calculate_pearson:
        pearson = pearsonr(x, y).statistic
        return data, pearson
    return data


def get_midpoint_scale(
    signal, midpoint_range, scale_range, num_trials, num_centerline_points
):
    def objective(trial):
        index = trial.suggest_float("index", midpoint_range[0], midpoint_range[1])
        scale = trial.suggest_float("scale", scale_range[0], scale_range[1])

        _, pearson = apply_midpoint_scale(
            signal, index, scale, num_centerline_points, calculate_pearson=True
        )
        return -pearson  # Optuna minimizes, so negate for maximization

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)

    best_params = study.best_params
    return best_params["index"], best_params["scale"]


def recenter(signals, skel, midpoint, scale, num_centerline_points):
    assert num_centerline_points % 2 == 0, "num_centerline_points must be even"
    signals = signals.copy()
    assert all([x not in signals for x in ["skel_z", "skel_y", "skel_x"]])
    signals["skel_z"] = skel[:, 0]
    signals["skel_y"] = skel[:, 1]
    signals["skel_x"] = skel[:, 2]

    signals = {
        key: apply_midpoint_scale(
            signals[key],
            midpoint,
            scale,
            num_centerline_points,
            calculate_pearson=False,
        )
        for key in signals
    }

    skel = np.stack(
        [
            signals.pop("skel_z"),
            signals.pop("skel_y"),
            signals.pop("skel_x"),
        ],
        axis=-1,
    )

    return signals, skel


# def normalize_signal(signal, midpoint):
#     I_baseline = signal[
#         int(signal.shape[0] * 0.45) : int(signal.shape[0] * 0.55)
#     ].mean()
#     if I_baseline == 0:
#         I_baseline = 1
#     return (signal - I_baseline) / I_baseline


def get_geodesic(skel, center):
    dist = np.cumsum(np.linalg.norm(np.diff(skel, axis=0), axis=1))
    dist = np.concatenate([[0], dist])
    dist -= dist[center]

    return dist


def _normalize_all_signals(
    skel,
    signals,
    extrapolate,
    gaussian_filter_ratio,
    midpoint_range,
    scale_range,
    num_trials,
    num_centerline_points,
):
    # calculate crop based on fiber channel
    start, stop, smoothed = get_tail_crop_points(
        signals["im_1"], extrapolate, gaussian_filter_ratio
    )

    skel = skel[start : stop + 1]
    signals = {k: v[start : stop + 1] for k, v in signals.items()}

    # midpoint based on im_2
    if "im_2" in signals:
        midpoint, scale = get_midpoint_scale(
            signals["im_2"],
            midpoint_range,
            scale_range,
            num_trials,
            num_centerline_points,
        )
    else:
        warnings.warn(
            "No 'im_2' signal found, using midpoint 0.5 and scale 0.0 as defaults."
        )
        midpoint = 0.5
        scale = 0.0
    # now apply these to all the signals
    signals, skel = recenter(signals, skel, midpoint, scale, num_centerline_points)

    geodesic = get_geodesic(skel, num_centerline_points // 2)
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
                        scale_range=conf.normalize_signals.scale_range,
                        num_trials=conf.normalize_signals.num_trials,
                        num_centerline_points=conf.skeletonize_fibers.num_centerline_points,
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
