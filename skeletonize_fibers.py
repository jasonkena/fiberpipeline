import os
import glob
import numpy as np
import h5py
import kimimaro
from em_util.io.box import compute_bbox_all
from tqdm import tqdm
from nicety.conf import get_conf
from joblib import Parallel, delayed


def skeletonize_chunk(vol, bbox_row, scale, const, anisotropy):
    fake_anisotropy = [8, 16, 16]
    vol = vol[
        bbox_row[1] : bbox_row[2] + 1,
        bbox_row[3] : bbox_row[4] + 1,
        bbox_row[5] : bbox_row[6] + 1,
    ]
    # have it a binary volume with value bbox_row[0]
    vol = (vol == bbox_row[0]) * bbox_row[0]
    skel = kimimaro.skeletonize(
        vol,
        teasar_params={
            "scale": scale,
            "const": const,
        },
        dust_threshold=0,  # manually filter based on length later
        anisotropy=fake_anisotropy,
    )
    assert len(skel) == 1, "Only one skeleton expected per chunk"
    skel = skel[bbox_row[0]]

    skel.vertices[:, 0] *= anisotropy[0] / fake_anisotropy[0]
    skel.vertices[:, 1] *= anisotropy[1] / fake_anisotropy[1]
    skel.vertices[:, 2] *= anisotropy[2] / fake_anisotropy[2]

    skel.vertices[:, 0] += bbox_row[1] * anisotropy[0]
    skel.vertices[:, 1] += bbox_row[3] * anisotropy[1]
    skel.vertices[:, 2] += bbox_row[5] * anisotropy[2]

    return skel


def generate_fiber_skeletons(conf, n_jobs=-1):
    files = sorted(glob.glob(os.path.join(conf.output_path, "*fiber_seg.h5")))

    for file in tqdm(files, desc="Processing files"):
        seg = h5py.File(file, "r")
        assert len(seg.keys()) == 1, "Only one key expected in segmentation file"
        seg = seg[list(seg.keys())[0]][:]
        # [id, z_min, z_max, y_min, y_max, x_min, x_max]
        # need to do z_min :  z_max + 1 to index
        bbox = compute_bbox_all(seg)

        import magicpickle

        magicpickle.send(
            (
                seg[
                    bbox[0, 1] : bbox[0, 2] + 1,
                    bbox[0, 3] : bbox[0, 4] + 1,
                    bbox[0, 5] : bbox[0, 6] + 1,
                ],
                bbox[0, 0],
            )
        )
        breakpoint()

        skels = list(
            tqdm(
                Parallel(n_jobs=n_jobs, return_as="generator")(
                    delayed(skeletonize_chunk)(
                        seg,
                        bbox[i],
                        conf.skeletonize_fibers.scale,
                        conf.skeletonize_fibers.const,
                        conf.anisotropy,
                    )
                    for i in range(bbox.shape[0])
                ),
                total=bbox.shape[0],
                leave=False,
            )
        )

        np.savez(
            os.path.join(
                conf.output_path,
                os.path.basename(file).replace("fiber_seg.h5", "fiber_skel.npz"),
            ),
            skels=skels,
        )


if __name__ == "__main__":
    conf = get_conf()
    generate_fiber_skeletons(conf)
