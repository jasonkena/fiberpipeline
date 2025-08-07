import os
import glob
import numpy as np
from pick import pick
import matplotlib.pyplot as plt

from nicety.conf import get_conf


def plot_signals(conf, basename):
    basename = ".".join(basename.split(".")[:-1])
    fname = os.path.join(conf.output_path, f"{basename}-final_signals.npz")
    signals = np.load(fname, allow_pickle=True)
    signals = signals["signals"].item()

    for i, signal_label in enumerate(signals.keys()):
        if not "im" in signal_label:
            print(f"Skipping {signal_label}")
            continue

        # Plot each skeleton's geodesic vs signal (no individual labels)
        for signal_vals, geodesic in zip(signals[signal_label], signals["geodesic"]):
            plt.plot(geodesic, signal_vals, alpha=0.3, linewidth=0.5)

        plt.xlabel("Geodesic Distance")
        plt.ylabel("Signal Value")
        plt.title(
            f"Geodesics vs Signal - {signal_label} (n={len(signals['geodesic'])} skeletons) for {basename}"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    conf = get_conf()

    title = "Select the dataset to plot"
    files = sorted(glob.glob(os.path.join(conf.dataset_path, "*.tif")))
    files = [os.path.basename(x) for x in files]

    _, index = pick(files, title, indicator="=>")
    print(f"Selected file: {files[index]}")

    plot_signals(conf, files[index])
