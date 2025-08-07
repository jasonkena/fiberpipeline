import os
import glob
import numpy as np
import h5py
from em_util.io.box import compute_bbox_all
from tqdm import tqdm
from nicety.conf import get_conf
from joblib import Parallel, delayed
from scipy.interpolate import splprep, splev, interp1d
from sklearn.decomposition import PCA
from cloudvolume import Skeleton
from typing import Optional, List


def fit_cylinder_spline_pca(
    points,
    manual_z_scale: float = 1.0,
    percentile_fit: List[float] = [0.0, 1.0],
    spline_smoothing: Optional[float] = None,
    npoints_geodesic: int = 1000,
):
    """
    Fit a spline to a deformed cylindrical point cloud using PCA coordinate system.
    Always uses arc length parameterization.

    Parameters:
    -----------
    points : array-like, shape (N, 3)
        Input point cloud coordinates
    percentile_fit : List[float]
        Percentiles to use for fitting the spline, e.g., [0.1, 0.9]
    spline_smoothing : Optional[float]
        Smoothing parameter for spline fitting (0 = interpolation)
    npoints_geodesic : int
        Number of points to sample along the geodesic for arc length computation

    Returns:
    --------
    evaluate_spline : callable
        Function to evaluate the fitted spline at given [0,1] arc length parameters
    total_length : float
    fp : float
        weighted sum of squared residuals of the spline fit

    """
    assert len(percentile_fit) == 2, "percentile_fit must contain exactly two values"

    # Step 0: Normalize points to unit sphere to allow spline_smoothing to take sane values
    points = points * np.array([manual_z_scale, 1.0, 1.0])
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))
    points = (points - center) / radius

    # Step 1: Fit PCA to find principal axes
    pca = PCA(n_components=3)
    pca.fit(points)

    # Transform points to PCA coordinate system
    points_pca = pca.transform(points)

    # Step 2: Use first PCA axis as parameter
    # Sort points by their position along the first PCA axis
    t_values = points_pca[:, 0]
    # since some t_values can be the same, need to pick unique values
    t_sorted, sorted_unique_indices = np.unique(t_values, return_index=True)

    # Sort all coordinates by the parameter
    points_pca_sorted = points_pca[sorted_unique_indices]

    # Normalize parameter to [0, 1]
    t_normalized = (t_sorted - t_sorted.min()) / (t_sorted.max() - t_sorted.min())

    used_to_fit = np.logical_and(
        t_normalized >= percentile_fit[0],
        t_normalized <= percentile_fit[1],
    )

    points_pca_sorted = points_pca_sorted[used_to_fit]
    t_normalized = t_normalized[used_to_fit]

    # Step 3: Fit spline in PCA coordinates using normalized parameter
    tck, u = splprep(
        [points_pca_sorted[:, 0], points_pca_sorted[:, 1], points_pca_sorted[:, 2]],
        u=t_normalized,
        s=spline_smoothing,
    )

    # Step 4: Arc length reparameterization
    # Sample the spline densely to compute arc length
    u_dense = np.linspace(0, 1, npoints_geodesic)
    spline_points_dense = np.array(splev(u_dense, tck)).T

    # Compute cumulative arc length
    diff = np.diff(spline_points_dense, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    arc_lengths = np.concatenate([[0], np.cumsum(distances)])

    # Normalize arc lengths to [0, 1]
    arc_lengths_normalized = arc_lengths / arc_lengths[-1]

    # Create inverse mapping: arc_length -> original_parameter
    arc_to_param = interp1d(
        arc_lengths_normalized,
        u_dense,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    # Create evaluation function
    def evaluate_spline(s_values):
        """
        Evaluate the spline at given arc length parameter values.

        Parameters:
        -----------
        s_values : array-like
            Arc length parameter values between 0 and 1

        Returns:
        --------
        points_xyz : array, shape (len(s_values), 3)
            Spline points in original XYZ coordinates
        """
        # Convert arc length parameters to original parameters
        u_values = arc_to_param(s_values)

        # Evaluate spline in PCA coordinates
        spline_pca = splev(u_values, tck)
        spline_pca = np.array(spline_pca).T

        # Transform back to original coordinates
        points_xyz = pca.inverse_transform(spline_pca)
        # Rescale to original space
        points_xyz = points_xyz * radius + center
        points_xyz = points_xyz * np.array([1 / manual_z_scale, 1.0, 1.0])

        return points_xyz

    return evaluate_spline, arc_lengths[-1]


def points_to_skeleton(points, segid: Optional[int] = None):
    vertices = points
    edges = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=int)
    skel = Skeleton(segid=segid, vertices=vertices, edges=edges)
    return skel


def skeletonize_chunk(
    vol,
    bbox_row,
    manual_z_scale,
    percentile_fit,
    spline_smoothing,
    num_centerline_points,
    extrapolate,
    anisotropy,
):
    vol = vol[
        bbox_row[1] : bbox_row[2] + 1,
        bbox_row[3] : bbox_row[4] + 1,
        bbox_row[5] : bbox_row[6] + 1,
    ]
    # have it a binary volume with value bbox_row[0]
    vol = vol == bbox_row[0]
    points = np.argwhere(vol) * np.array(anisotropy)
    assert len(points) > 0, "No points found in the volume"

    spline, total_length = fit_cylinder_spline_pca(
        points,
        manual_z_scale=manual_z_scale,
        percentile_fit=percentile_fit,
        spline_smoothing=spline_smoothing,
    )
    centerline = spline(
        np.linspace(extrapolate[0], extrapolate[1], num_centerline_points)
    )
    centerline[:, 0] += bbox_row[1] * anisotropy[0]
    centerline[:, 1] += bbox_row[3] * anisotropy[1]
    centerline[:, 2] += bbox_row[5] * anisotropy[2]

    skel = points_to_skeleton(centerline, segid=bbox_row[0])

    return bbox_row[0], skel


def generate_fiber_skeletons(conf, n_jobs=-1):
    files = sorted(glob.glob(os.path.join(conf.output_path, "*-fiber_seg.h5")))

    for file in tqdm(files, desc="Processing files"):
        seg = h5py.File(file, "r")
        assert len(seg.keys()) == 1, "Only one key expected in segmentation file"
        seg = seg[list(seg.keys())[0]][:]
        bbox = compute_bbox_all(seg)

        skels = list(
            tqdm(
                Parallel(n_jobs=n_jobs, return_as="generator")(
                    delayed(skeletonize_chunk)(
                        seg,
                        bbox[i],
                        conf.skeletonize_fibers.manual_z_scale,
                        conf.skeletonize_fibers.percentile_fit,
                        conf.skeletonize_fibers.spline_smoothing,
                        conf.skeletonize_fibers.num_centerline_points,
                        conf.skeletonize_fibers.extrapolate,
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
            skel_ids=list(map(lambda x: x[0], skels)),
            skels=list(map(lambda x: x[1], skels)),
        )


if __name__ == "__main__":
    conf = get_conf()
    generate_fiber_skeletons(conf)
