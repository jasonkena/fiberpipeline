import os
import numpy as np
from tqdm import tqdm
from glob import glob
from scipy.stats import mode

from fiberpipeline.utils import get_basename
from nicety.conf import get_conf

from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from collections import defaultdict


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


def get_valid(
    skel_ids,
    lengths,
    pca_ratios,
    mean_somas,
    cell_labels,
    thres_length,
    thres_pca_ratio,
    thres_mean_soma,
    one_per_soma,
):
    length_valid = lengths > thres_length
    print(f"Length valid: {np.sum(length_valid)}/{len(length_valid)} fibers")
    pca_valid = pca_ratios > thres_pca_ratio
    print(f"PCA valid: {np.sum(pca_valid)}/{len(pca_valid)} fibers")
    soma_valid = mean_somas > thres_mean_soma
    print(f"Mean soma valid: {np.sum(soma_valid)}/{len(soma_valid)} fibers")
    in_soma_valid = cell_labels != 0
    print(f"In soma valid: {np.sum(in_soma_valid)}/{len(in_soma_valid)} fibers")

    is_valid = length_valid & pca_valid & soma_valid & in_soma_valid
    print(f"Valid fibers: {np.sum(is_valid)}/{len(is_valid)} fibers")

    if one_per_soma:
        cell_label_to_skel_id = defaultdict(list)
        skel_id_to_idx = {}

        for i, (skel_id, cell_id, is_valid_fiber) in enumerate(
            zip(skel_ids, cell_labels, is_valid)
        ):
            skel_id_to_idx[skel_id] = i
            if is_valid_fiber:
                cell_label_to_skel_id[cell_id].append(skel_id)
        cell_label_to_skel_id = {
            cell_id: max(ids, key=lambda x: lengths[skel_id_to_idx[x]])
            for cell_id, ids in cell_label_to_skel_id.items()
        }
        final_valid = np.zeros(len(skel_ids), dtype=bool)
        for x in cell_label_to_skel_id.values():
            final_valid[skel_id_to_idx[x]] = True

        print(
            f"Final valid fibers: {np.sum(final_valid)}/{np.sum(is_valid)} representatives picked"
        )
    else:
        final_valid = is_valid

    return final_valid


def validate_fibers(conf):
    output_path = conf.output_path
    basenames = sorted(
        [
            get_basename(fname)
            for fname in glob(f"{output_path}/*-normalized_signals.npz")
        ]
    )

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

        is_valid = get_valid(
            skel_ids,
            lengths,
            pca_ratios,
            mean_somas,
            cell_labels,
            conf.validate_fibers.thres_length,
            conf.validate_fibers.thres_pca_ratio,
            conf.validate_fibers.thres_mean_soma,
            conf.validate_fibers.one_per_soma,
        )

        np.savez(
            os.path.join(
                conf.output_path,
                basename + "-final_signals.npz",
            ),
            # ['im_0', 'im_1', 'im_2', 'im_3', 'fiber_seg', 'cell_seg', 'geodesic']
            signals={name: np.array(signals[name]) for name in signals},
            # list of [Nx3] arrays
            skels=[np.array(skel) for skel in skels],
            skel_ids=skel_ids,
            criteria={
                "lengths": lengths,
                "pca_ratios": pca_ratios,
                "mean_somas": mean_somas,
                "cell_labels": cell_labels,
            },
            is_valid=is_valid,
        )


if __name__ == "__main__":
    conf = get_conf()
    validate_fibers(conf)
