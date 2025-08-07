import cv2
from nicety.conf import get_conf
from glob import glob
from tqdm import tqdm
from connectomics.data.utils.data_io import readvol, savevol


def apply_tiled_clahe(vol, clip_limit=2.0, tile_grid_size=(8, 8), kernel_size=256):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    for z in range(vol.shape[0]):
        for y in range(0, vol.shape[1], kernel_size):
            for x in range(0, vol.shape[2], kernel_size):
                y_max = min(y + kernel_size, vol.shape[1])
                x_max = min(x + kernel_size, vol.shape[2])
                vol[z, y:y_max, x:x_max] = clahe.apply(vol[z, y:y_max, x:x_max])
    return vol


if __name__ == "__main__":
    conf = get_conf()
    data_dir = conf.fiber_segmentation_model.dataset_path

    for fname in (pbar := tqdm(sorted(glob(f"{data_dir}/*")))):
        pbar.set_description(fname)
        data = readvol(fname)
        data = apply_tiled_clahe(
            data,
            clip_limit=conf.clahe.clip_limit,
            tile_grid_size=conf.clahe.tile_grid_size,
            kernel_size=conf.clahe.kernel_size,
        )
        savevol(fname.replace(".", "-tiled_clahe."), data)
