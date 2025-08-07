import os
import glob
from tqdm import tqdm
import imageio
import h5py

from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)
from nicety.conf import get_conf
from fiberpipeline.utils import preprocess_vol, get_basename

from tqdm import tqdm


def segment(vol, model_type):
    predictor, segmenter = get_predictor_and_segmenter(model_type)
    seg = automatic_instance_segmentation(
        predictor, segmenter, input_path=vol, verbose=False
    )
    return seg


def main(conf):
    dataset_path = conf.dataset_path
    os.makedirs(conf.output_path, exist_ok=True)

    files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))
    for file in (pbar := tqdm(files)):
        pbar.set_description(get_basename(file))

        vol = imageio.volread(file)
        vol = preprocess_vol(vol)  # C, Z, Y, X
        vol = vol[0]  # cell channel
        seg = segment(vol, conf.cell_segmentation.model_type)

        file = h5py.File(
            os.path.join(
                conf.output_path, os.path.basename(file).replace(".tif", "-cell_seg.h5")
            ),
            "w",
        )
        file.create_dataset("main", data=seg, compression="gzip")


if __name__ == "__main__":
    conf = get_conf()
    main(conf)
