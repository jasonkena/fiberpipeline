import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from nicety.conf import get_conf

def plot_signals(conf):
    files = sorted(glob.glob(os.path.join(conf.output_path, "*-processed_signals.npz")))
    for file in files:
        print(f"Processing {file}")
        signals = np.load(file, allow_pickle=True)
        signals, signal_labels, geodesics = signals["signals"], signals["signal_labels"], signals['geodesics']
        skels = np.load(
            file.replace("processed_signals.npz", "fiber_skel.npz"), allow_pickle=True
        )["skels"].tolist()

        thres = 250
        mask = [True if g[-1] - g[0] < thres else False for g in geodesics]

        signals = np.delete(signals, mask, axis=1)
        geodesics = np.delete(geodesics, mask, axis=0)
        skels = np.delete(skels, mask, axis=0)

        for i, signal_label in enumerate(signal_labels):
            if not "im" in signal_label:
                print(f"Skipping {signal_label}")
                continue

            # Plot each skeleton's geodesic vs signal (no individual labels)
            for signal_vals, geodesic in zip(
                signals[i], geodesics
            ):
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
