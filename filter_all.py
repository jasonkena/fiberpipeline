# suppress warnings for outdated protobuf version
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

    import os
    import cc3d
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    
    import dill
    from connectomics.data.utils.data_io import readvol, savevol
    from multiprocessing import Pool

    from utils import get_basename
    from nicety.conf import get_conf

def filter_by_length(conf, signals, geodesics, skels, skel_ids):
    mask = [False if g[-1] - g[0] >= conf.filter_all.thres_length else True for g in geodesics]
    signals = np.delete(signals, mask, axis=1)
    geodesics = np.delete(geodesics, mask, axis=0)
    skels = np.delete(skels, mask, axis=0)
    skel_ids = np.delete(skel_ids, mask, axis=0)
    return signals, geodesics, skels, skel_ids

def skel_to_dist_mean(skel):
    p0, p1 = skel.vertices[0], skel.vertices[-1]
    dists = list(map(lambda x: np.linalg.norm(np.cross(p1-p0,x-p0)/np.linalg.norm(p1-p0)), skel.vertices))
    return np.mean(dists)

def filter_by_shape(conf, signals, geodesics, skels, skel_ids):
    with Pool(conf.num_cpus) as p:
        means = list(p.map(skel_to_dist_mean, skels))
    mask = [False if m < (g[-1] - g[0]) * conf.filter_all.thres_shape else True for m, g in zip(means, geodesics)]
    signals = np.delete(signals, mask, axis=1)
    geodesics = np.delete(geodesics, mask, axis=0)
    skels = np.delete(skels, mask, axis=0)
    skel_ids = np.delete(skel_ids, mask, axis=0)
    return signals, geodesics, skels, skel_ids

def filter_by_cell(conf, signals, geodesics, skels, skel_ids, fiber_seg, cell_seg):
    cell_seg = np.clip(cell_seg, 0, 1).astype(np.uint8)
    allow_list, counts = np.unique(fiber_seg * cell_seg, return_counts=True)
    allow_list = np.array([idx for idx, count in zip(allow_list, counts) if count > 300])
    mask = [False if idx in allow_list else True for idx in skel_ids]
    signals = np.delete(signals, mask, axis=1)
    geodesics = np.delete(geodesics, mask, axis=0)
    skels = np.delete(skels, mask, axis=0)
    skel_ids = np.delete(skel_ids, mask, axis=0)
    return signals, geodesics, skels, skel_ids

def cubeness_metric(side_lengths):
    return 1 - np.std(side_lengths) / np.mean(side_lengths)

def filter_by_ratio(conf, signals, geodesics, skels, skel_ids):
    bbox_dims = list(map(lambda skel: np.max(skel.vertices, axis=0) - np.min(skel.vertices, axis=0), skels))
    with Pool(conf.num_cpus) as p:
        mask = [False if cubeness > conf.filter_all.thres_ratio else True for cubeness in list(p.map(cubeness_metric, bbox_dims))]
    signals = np.delete(signals, mask, axis=1)
    geodesics = np.delete(geodesics, mask, axis=0)
    skels = np.delete(skels, mask, axis=0)
    skel_ids = np.delete(skel_ids, mask, axis=0)
    return signals, geodesics, skels, skel_ids

if __name__ == '__main__':
    conf = get_conf()
    output_path = conf.output_path
    extensions = ['.npz', '.h5']
    basenames = np.unique(sorted([get_basename(fname) for fname in glob(f'{output_path}/*') if os.path.splitext(fname)[1] in extensions]))

    for basename in (pbar := tqdm(basenames)):
        pbar.set_description(basename)
        
        signals = np.load(os.path.join(output_path, basename + '-normalized_signals.npz'), allow_pickle=True)
        signals, signal_labels, geodesics = signals['signals'], signals['signal_labels'], signals['geodesics']
        skels = np.load(os.path.join(output_path, basename + '-fiber_skel.npz'), allow_pickle=True)
        skels, skel_ids = skels['skels'], skels['skel_ids']
        fiber_seg = readvol(os.path.join(output_path, basename + '-fiber_seg.h5'))
        cell_seg = readvol(os.path.join(output_path, basename + '-cell_seg.h5'))

        #signals, geodesics, skels, skel_ids = filter_by_length(conf, signals, geodesics, skels, skel_ids)
        #signals, geodesics, skels, skel_ids = filter_by_shape(conf, signals, geodesics, skels, skel_ids)
        #signals, geodesics, skels, skel_ids = filter_by_cell(conf, signals, geodesics, skels, skel_ids, fiber_seg, cell_seg)
        signals, geodesics, skels, skel_ids = filter_by_ratio(conf, signals, geodesics, skels, skel_ids)

        np.savez(
            os.path.join(
                conf.output_path,
                basename + "-filtered_signals.npz",
            ),
            signals=signals,
            signal_labels=signal_labels,
            geodesics=geodesics
        )

        np.savez(
            os.path.join(
                conf.output_path,
                basename + "-filtered_fiber_skel.npz",
            ),
            skels=skels,
            skel_ids=skel_ids
        )
