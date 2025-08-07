import os
import numpy as np
import imageio
import h5py
import glob
from nicety.conf import get_conf, dotdict_to_dict
from tqdm import tqdm

from fiberpipeline.utils import preprocess_vol
import cv2
from joblib import Parallel, delayed
from omegaconf import OmegaConf

import tempfile

from connectomics.utils.system import init_devices
from connectomics.config import load_cfg
from connectomics.engine import Trainer
from connectomics.utils.process import remove_small_instances, cast2dtype
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.transform import resize

import argparse


def clahe_slice(slice_2d, tile_size, clip_limit):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(slice_2d)


def clahe_parallel(vol, tile_size=8, clip_limit=2.0, n_jobs=-1):
    slices = list(
        tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(clahe_slice)(vol[i], tile_size=tile_size, clip_limit=clip_limit)
                for i in range(vol.shape[0])
            ),
            total=vol.shape[0],
            leave=False,
        )
    )

    return np.stack(slices, axis=0)


def pytc_main(pytc_yaml_path, pytc_checkpoint_path):
    args = argparse.Namespace(
        config_file=pytc_yaml_path,
        config_base=None,
        inference=True,
        distributed=False,
        checkpoint=pytc_checkpoint_path,
        manual_seed=None,
        local_world_size=1,
        local_rank=None,
        debug=False,
        opts=[],
    )
    cfg = load_cfg(args)
    device = init_devices(args, cfg)

    # start training or inference
    mode = "test" if args.inference else "train"
    trainer = Trainer(
        cfg, device, mode, rank=args.local_rank, checkpoint=args.checkpoint
    )

    # Start training or inference:
    if cfg.DATASET.DO_CHUNK_TITLE == 0:
        test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
        test_func() if args.inference else trainer.train()
    else:
        trainer.run_chunk(mode)

    print("Rank: {}. Device: {}. Process is finished!".format(args.local_rank, device))


def generate_affinities(conf):
    dataset_path = conf.dataset_path

    os.makedirs(conf.output_path, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=conf.output_path) as tmpdir:
        files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))
        for file in tqdm(files, desc="Processing files"):
            # C, Z, Y, X
            img = preprocess_vol(imageio.volread(file))

            img = clahe_parallel(
                img[1],  # fiber channel
                tile_size=conf.fiber_segmentation.clahe.tile_size,
                clip_limit=conf.fiber_segmentation.clahe.clip_limit,
            )  # Apply CLAHE to the first channel
            # save to tmpdir
            imageio.volwrite(os.path.join(tmpdir, "inference.tif"), img)

            pytc_yaml = get_conf(
                [conf.fiber_segmentation.base_yaml, conf.fiber_segmentation.bcs_yaml]
            )
            pytc_yaml.SYSTEM.NUM_GPUS = conf.num_gpus
            pytc_yaml.SYSTEM.NUM_CPUS = conf.num_cpus
            pytc_yaml.INFERENCE.INPUT_PATH = tmpdir
            pytc_yaml.INFERENCE.IMAGE_NAME = "inference.tif"
            pytc_yaml.INFERENCE.OUTPUT_PATH = tmpdir
            pytc_yaml.INFERENCE.OUTPUT_NAME = "affinities.h5"
            pytc_yaml.INFERENCE.SAMPLES_PER_BATCH = conf.fiber_segmentation.batch_size
            pytc_yaml = dotdict_to_dict(pytc_yaml)

            pytc_yaml_path = os.path.join(tmpdir, "pytc.yaml")
            OmegaConf.save(pytc_yaml, pytc_yaml_path)
            pytc_main(pytc_yaml_path, conf.fiber_segmentation.checkpoint)

            os.rename(
                os.path.join(tmpdir, "affinities.h5"),
                os.path.join(
                    conf.output_path,
                    os.path.basename(file).replace(".tif", "-fiber_aff.h5"),
                ),
            )


def bcs_watershed(
    volume,
    thres1=0.9,
    thres2=0.8,
    thres3=0.85,
    # thres4=0.80,
    thres4=0.50,
    thres5=-1.0,
    thres_small=128,
    scale_factors=(1.0, 1.0, 1.0),
    remove_small_mode="background",
    seed_thres=32,
    return_seed=False,
    precomputed_seed=None,
):
    # adapted from https://github.com/jasonkena/pytorch_connectomics/blob/f29a6bf71b2d82171392a6b69cec37fa6c898f92/connectomics/utils/process.py#L171
    r"""Convert binary foreground probability maps, instance contours and skeleton-aware distance
    transform to instance masks via watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.9
        thres2 (float): threshold of instance contours. Default: 0.8
        thres3 (float): threshold of foreground. Default: 0.85
        thres4 (float): threshold of signed distance for locating seeds. Default: 0.5
        thres5 (float): threshold of signed distance for foreground. Default: 0.0
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
    """
    assert volume.shape[0] == 3
    semantic, boundary, distance = volume[0], volume[1], volume[2]
    distance = distance / 255.0
    foreground = (semantic > int(255 * thres3)) * (distance > thres5)

    if precomputed_seed is not None:
        seed = precomputed_seed
    else:  # compute the instance seeds
        seed_map = (
            (semantic > int(255 * thres1))
            * (boundary < int(255 * thres2))
            * (distance > thres4)
        )
        seed = label(seed_map)
        seed = remove_small_objects(seed, seed_thres)

    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x == 1.0 for x in scale_factors):
        target_size = (
            int(semantic.shape[0] * scale_factors[0]),
            int(semantic.shape[1] * scale_factors[1]),
            int(semantic.shape[2] * scale_factors[2]),
        )
        segm = resize(
            segm, target_size, order=0, anti_aliasing=False, preserve_range=True
        )

    if not return_seed:
        return cast2dtype(segm)

    return cast2dtype(segm), seed


def generate_segmentation(conf):
    bcs_conf = conf.fiber_segmentation.bcs

    assert os.path.exists(conf.output_path), "Output path does not exist"

    files = sorted(glob.glob(os.path.join(conf.output_path, "*fiber_aff.h5")))

    for file in tqdm(files, desc="Processing files"):
        aff = h5py.File(file, "r")
        assert len(aff.keys()) == 1, "Only one key expected in affinity file"
        aff = aff[list(aff.keys())[0]][:]
        seg = bcs_watershed(
            aff,
            thres1=bcs_conf.thres1,
            thres2=bcs_conf.thres2,
            thres3=bcs_conf.thres3,
            thres4=bcs_conf.thres4,
            thres5=bcs_conf.thres5,
            thres_small=bcs_conf.thres_small,
        )
        h5_file = h5py.File(
            os.path.join(
                conf.output_path,
                os.path.basename(file).replace("fiber_aff.h5", "fiber_seg.h5"),
            ),
            "w",
        )
        h5_file.create_dataset("main", data=seg, compression="gzip")


if __name__ == "__main__":
    conf = get_conf()
    generate_affinities(conf)
    generate_segmentation(conf)
