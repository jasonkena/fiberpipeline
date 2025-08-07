import os
import h5py
import glob
import numpy as np
from nicety.conf import get_conf
from fiberpipeline.utils import get_basename
from tqdm import tqdm


def stats(conf):
    basenames = sorted(
        [
            get_basename(f)
            for f in glob.glob(os.path.join(conf.output_path, "*-final_signals.npz"))
        ]
    )

    stats = {}
    for basename in tqdm(basenames):
        data = np.load(
            os.path.join(conf.output_path, f"{basename}-final_signals.npz"),
            allow_pickle=True,
        )
        signals_dict = data["signals"].item()
        criteria = data["criteria"].item()
        lengths = criteria["lengths"]
        cell_labels = criteria["cell_labels"]
        # strip those not in cell and too short
        cell_labels = cell_labels[
            (cell_labels > 0) & (lengths > conf.validate_fibers.thres_length)
        ]
        unique_cell_labels, counts_per_soma = np.unique(cell_labels, return_counts=True)

        mean_counts_per_soma = np.mean(counts_per_soma)

        unique_counts, frequency = np.unique(counts_per_soma, return_counts=True)

        pca_ratios = np.array(criteria["pca_ratios"])  # shape: (n, 2)
        im0 = signals_dict["im_0"]
        im1 = signals_dict["im_1"]
        im2 = signals_dict["im_2"]
        im3 = signals_dict["im_3"]
        cond_length = (lengths > 800) & (lengths < 2000)
        cond_min_im0 = np.min(im0, axis=1) > 100
        cond_min_im1 = np.min(im1, axis=1) > 200
        cond_min_im2 = np.min(im2, axis=1) > 20
        cond_min_im3 = np.min(im3, axis=1) > 20
        cond_pca = pca_ratios > 0.8
        cond_high_im0 = (np.sum(im0 > 500, axis=1) / im0.shape[1]) > 0.1
        final_valid = (
            cond_length
            & cond_min_im0
            & cond_min_im1
            & cond_min_im2
            & cond_min_im3
            & cond_pca
            & cond_high_im0
            & (criteria["cell_labels"] > 0)
        )
        valid_cells = np.unique(criteria["cell_labels"][final_valid])

        # cell_seg = h5py.File(
        #     os.path.join(conf.output_path, f"{basename}-cell_seg.h5"), "r"
        # )["main"]
        stats[basename] = {
            # "soma_count": int(np.max(cell_seg)),
            "unique_counts": unique_counts,
            "frequency": frequency,
            "mean_counts_per_soma": mean_counts_per_soma,
            "num_fibers": len(lengths),
            "final_valid": final_valid,
            "valid_cells": valid_cells,
        }
    for basename in stats:
        print(f"{basename}: {len(stats[basename]['valid_cells'])} valid cells")
        # print(f"{basename}: {np.sum(stats[basename]['final_valid'])} fibers after final filter")
        # print(f"{basename}: {stats[basename]['num_fibers']} fibers")
    # print(f"Total fibers: {sum(s['num_fibers'] for s in stats.values())}")
    np.savez(
        os.path.join(conf.output_path, "stats.npz"),
        stats=stats,
    )


if __name__ == "__main__":
    conf = get_conf()
    stats(conf)
