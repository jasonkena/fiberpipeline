import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from nicety.conf import get_conf


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
        signals, signal_labels, midpoints = signals["signals"], signals["signal_labels"], signals["midpoints"]
        skels = np.load(
            file.replace("_signals.npz", "_fiber_skel.npz"), allow_pickle=True
        )["skels"].tolist()
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
