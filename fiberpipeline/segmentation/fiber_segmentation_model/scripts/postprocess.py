import numpy as np
from connectomics.data.utils.data_io import readvol, savevol
from connectomics.utils.evaluate import adapted_rand
from connectomics.utils.process import remove_small_instances, cast2dtype
from skimage import exposure

from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.transform import resize

from nicety.conf import get_conf
from argparse import ArgumentParser
from tqdm import tqdm
import sys

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical


def rescale(image, percentile, out_range):
    """
    Rescale image intensity to the given percentile range.
    """
    image = image.astype(np.float32)
    p0, p1 = np.percentile(image, percentile)
    return exposure.rescale_intensity(image, in_range=(p0, p1), out_range=out_range)


def bcs_watershed(
    volume,
    thres1=0.9,
    thres2=0.8,
    thres3=0.85,
    thres4=0.80,
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


def process_bcs(input_path, output_path):
    for fname in (pbar := tqdm.tqdm(sorted(glob.glob(f"{input_path}/*.h5")))):
        pbar.set_description(fname)
        data = readvol(input_path)
        # rescale skel channel
        data[-1] = rescale(
            data[-1],
            tuple(conf.fiber_segmentation.rescale_params.percentile),
            tuple(conf.fiber_segmentation.rescale_params.out_range),
        )
        data = bcs_watershed(
            data,
            thres1=conf.fiber_segmentation.bcs.thres1,
            thres2=conf.fiber_segmentation.bcs.thres2,
            thres3=conf.fiber_segmentation.bcs.thres3,
            thres4=conf.fiber_segmentation.bcs.thres4,
            thres5=conf.fiber_segmentation.bcs.thres5,
            thres_small=conf.fiber_segmentation.bcs.thres_small,
        )
        nname = fname.replace(".", "-seg.")
        savevol(os.path.join(output_path, nname), data)


def arand_wrapper(params):
    thres1, thres2, thres3, thres4, thres5, thres_small, pred, gt = params
    pred = readvol(pred)
    gt = readvol(gt)
    bcs_seg = bcs_watershed(
        pred,
        thres1=thres1,
        thres2=thres2,
        thres3=thres3,
        thres4=thres4,
        thres5=thres5,
        thres_small=thres_small,
    )
    return adapted_rand(bcs_seg, gt)


def optimize_bcs(pred_path, gt_path):
    dimensions = [
        (0.0, 1.0),  # thres1
        (0.0, 1.0),  # thres2
        (0.0, 1.0),  # thres3
        (0.0, 1.0),  # thres4
        (0.0, 1.0),  # thres5
        (10, 1000),  # thres_small
        [pred_path],
        [gt_path],
    ]
    res = gp_minimize(func=arand_wrapper, dimensions=dimensions, n_calls=100)
    print(f"x: {res.x[:6]}\nres: {res.fun}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        action="append",
        choices=["bcs_watershed", "optimize_bcs"],
        help="Task to run",
        required=True,
    )
    parser.add_argument(
        "--input",
        action="append",
        help="Input directory path for bcs_watershed, three-channel affinity map produced from model for optimize_bcs",
        required=True,
    )
    parser.add_argument(
        "--output",
        action="append",
        help="Output directory path for bcs_watershed, ground truth path in optimize_bcs",
        required=True,
    )
    args = parser.parse_args()
    task, in_file, out_file = args.task[0], args.input[0], args.output[0]
    conf = get_conf()
    if task == "bcs_watershed":
        bcs_watershed(in_file, out_file)
    if task == "optimize_bcs":
        optimize_bcs(in_file, out_file)
