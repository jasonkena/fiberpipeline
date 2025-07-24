import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from nicety.conf import get_conf


def plot_signals(conf):
    files = sorted(glob.glob(os.path.join(conf.output_path, "*-filtered_signals.npz")))
    for file in files:
        print(f"Processing {file}")
        signals = np.load(file, allow_pickle=True)
        signals, signal_labels, geodesics = (
            signals["signals"],
            signals["signal_labels"],
            signals["geodesics"],
        )
        skels = np.load(
            file.replace("filtered_signals", "filtered_fiber_skel"), allow_pickle=True
        )["skels"].tolist()

        for i, signal_label in enumerate(signal_labels):
            if not "im" in signal_label:
                print(f"Skipping {signal_label}")
                continue

            # Plot each skeleton's geodesic vs signal (no individual labels)
            for signal_vals, geodesic in zip(signals[i], geodesics):
                plt.plot(geodesic, signal_vals, alpha=0.3, linewidth=0.5)

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
