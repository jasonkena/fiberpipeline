import os
import numpy as np
from tqdm import tqdm
from glob import glob
from scipy.stats import mode

from utils import get_basename
from nicety.conf import get_conf

from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA


def evaluate_fiber(signals, skel):
    # geodesic distance
    length = signals["geodesic"][-1] - signals["geodesic"][0]

    # calculate PCA
    center = np.mean(skel, axis=0)
    radius = np.max(np.linalg.norm(skel - center, axis=1))
    points = (skel - center) / radius
    pca = PCA(n_components=3)
    pca.fit(points)
    pca_ratio = pca.explained_variance_ratio_[0]

    # mean brightness of soma
    mean_soma = np.mean(signals["im_0"])

    # calculate most common cell label
    cell_label = mode(signals["cell_seg"]).mode

    return length, pca_ratio, mean_soma, cell_label


def filter_fibers(conf):
    output_path = conf.output_path
    extensions = [".npz", ".h5"]
    basenames = np.unique(
        sorted(
            [
                get_basename(fname)
                for fname in glob(f"{output_path}/*")
                if os.path.splitext(fname)[1] in extensions
            ]
        )
    )

    diff_list = []

    for basename in (pbar := tqdm(basenames)):
        pbar.set_description(basename)

        signals = np.load(
            os.path.join(output_path, basename + "-normalized_signals.npz"),
            allow_pickle=True,
        )
        # ['im_0', 'im_1', 'im_2', 'im_3', 'fiber_seg', 'cell_seg', 'geodesic']
        signals, skels, skel_ids = (
            signals["signals"].item(),
            signals["skels"].tolist(),
            signals["skel_ids"],
        )

        res = list(
            tqdm(
                Parallel(n_jobs=conf.num_cpus, return_as="generator")(
                    delayed(evaluate_fiber)(
                        {name: signals[name][i] for name in signals},
                        skels[i],
                    )
                    for i in range(len(skels))
                ),
                total=len(skels),
                leave=False,
            )
        )
        lengths = np.array([x[0] for x in res])
        pca_ratios = np.array([x[1] for x in res])
        mean_somas = np.array([x[2] for x in res])
        cell_labels = np.array([x[3] for x in res])

        np.savez(
            os.path.join(
                conf.output_path,
                basename + "-fiber_attributes.npz",
            ),
            lengths=lengths,
            pca_ratios=pca_ratios,
            mean_somas=mean_somas,
            cell_labels=cell_labels,
        )

        # np.savez(
        #     os.path.join(
        #         conf.output_path,
        #         basename + "-filtered_fiber_skel.npz",
        #     ),
        #     skels=skels,
        #     skel_ids=skel_ids,
        # )
        #


if __name__ == "__main__":
    conf = get_conf()
    filter_fibers(conf)
