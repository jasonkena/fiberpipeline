import os
import cc3d
import numpy as np
from tqdm import tqdm
from glob import glob

from connectomics.data.utils.data_io import readvol, savevol

from nicety.conf import get_conf

def mask_fibers():
    for fiber_fname in (pbar := tqdm(sorted(glob(os.path.join(conf.output_path, '*fiber_seg.h5'))))):
        basename = os.path.basename(fiber_fname).replace('-fiber_seg', '')
        pbar.set_description(basename)
        cell_fname = fiber_fname.replace('fiber_seg', 'cell_seg')
        
        cell = np.clip(readvol(cell_fname), 0, 1)
        fiber = readvol(fiber_fname) * cell
        
        fiber = cc3d.dust(fiber, threshold=100)
        savevol(fiber_fname.replace('fiber_seg', 'masked_fiber_seg'), fiber)

if __name__ == '__main__':
    conf = get_conf()
    mask_fibers()
