import glob
import os
import socketserver
import threading
import tempfile
from urllib.parse import urlparse
from typing import Dict, Tuple

from cloudvolume import CloudVolume
from cloudvolume.server import ViewerServerHandler

import napari
from napari import Viewer
import fastremap

import numpy as np
import neuroglancer
import imageio
import h5py
from pick import pick

from nicety.conf import get_conf, DotDict
from fiberpipeline.utils import preprocess_vol
from fiberpipeline.plot.precomputed import to_precomputed, write_skeletons
from fiberpipeline.processing.validate_fibers import get_valid
from fiberpipeline.processing.skeletonize_fibers import points_to_skeleton


def get_free_port(ip: str) -> int:
    # https://stackoverflow.com/a/61685162/10702372
    with socketserver.TCPServer((ip, 0), None) as s:
        return s.server_address[1]


def serve_directory(ip: str, directory: str, port: int):
    def handler(*args, **kwargs):
        return ViewerServerHandler(directory, *args, **kwargs)

    with socketserver.TCPServer((ip, port), handler) as httpd:
        print(f"Serving {directory} at http://localhost:{port}")
        httpd.serve_forever()


def serve_cloudvolume_layers(ip: str, layers: Dict[str, CloudVolume]) -> DotDict:
    layerpaths = {name: cv.layerpath for name, cv in layers.items()}
    common_path = os.path.commonpath(list(layerpaths.values()))
    relative_paths = {
        name: os.path.relpath(path, common_path) for name, path in layerpaths.items()
    }

    port = get_free_port(ip)
    thread = threading.Thread(
        target=serve_directory,
        args=(ip, common_path, port),
        daemon=True,
    )
    thread.start()

    sources = {
        name: f"precomputed://http://{ip}:{port}/{path}"
        for name, path in relative_paths.items()
    }

    return port, thread, sources


def plot_neuroglancer(layers: Dict[str, CloudVolume], im_shader: str):
    ip = "localhost"
    neuroglancer.set_server_bind_address(bind_address=ip)
    viewer = neuroglancer.Viewer()

    port, thread, sources = serve_cloudvolume_layers(ip, layers)
    ports = [port]

    with viewer.txn() as s:
        for name in layers.keys():
            assert layers[name].layer_type in ["image", "segmentation"]

            if layers[name].layer_type == "image":
                s.layers.append(
                    name=name,
                    layer=neuroglancer.ImageLayer(
                        source=sources[name],
                        shader=im_shader,
                        volume_rendering_mode="max",
                    ),
                )
            else:
                cv = layers[name]
                s.layers.append(
                    name=name,
                    layer=neuroglancer.SegmentationLayer(
                        source=sources[name],
                    ),
                    # tell NG which segments exist
                    segments=sorted(map(int, cv.image.unique(cv.bounds, cv.mip))),
                )

    print(viewer)
    ports.append(urlparse(str(viewer)).port)
    ports = " ".join(map(str, ports))
    print(f"Ports: {ports}")

    return viewer


def plot_napari(
    layers: Dict[str, Tuple[np.ndarray, str]],
    anisotropy,
    to_um,
    fiber_skels=None,
    fiber_skel_ids=None,
    skel_downsample=100,
    link_fiber_skels_to: str = "fiber_seg",
):
    """
    Plot layers in napari.
    """
    viewer = Viewer()
    scale = [x * to_um for x in anisotropy]
    for name, (vol, layer_type) in layers.items():
        assert layer_type in ["image", "segmentation"]

        vol = vol[:]
        if layer_type == "image":
            viewer.add_image(
                vol,
                name=name,
                scale=scale,
                blending="translucent_no_depth",
            )
        else:
            viewer.add_labels(
                vol,
                name=name,
                scale=scale,
                rendering="translucent",
            )

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    if fiber_skels is not None:
        assert fiber_skel_ids is not None
        # Flatten all skeleton vertices into a single array
        all_points = []
        point_skel_ids = []
        colors = []

        for i, skel in enumerate(fiber_skels):
            skel_id = fiber_skel_ids[i]
            points = skel[::skel_downsample] * to_um
            all_points.append(points)
            # Create skel_id for each point in this skeleton
            point_skel_ids.extend([skel_id] * len(points))
            colors.append(
                [viewer.layers[link_fiber_skels_to].get_color(skel_id)] * len(points)
            )

        # Concatenate all points and convert skel_ids to numpy array
        all_points = np.concatenate(all_points, axis=0)
        point_skel_ids = np.array(point_skel_ids)
        colors = np.concatenate(colors, axis=0)

        viewer.add_points(
            all_points,
            blending="translucent_no_depth",
            name="fiber_skeletons",
            # Color points by skeleton ID
            face_color=colors,
            border_width=0,
            size=0.25,  # Adjust point size as needed
        )
    napari.run()

    return viewer


if __name__ == "__main__":
    conf = get_conf()

    title = "Select the dataset to plot"
    files = sorted(glob.glob(os.path.join(conf.dataset_path, "*.tif")))
    files = [os.path.basename(x) for x in files]

    _, index = pick(files, title, indicator="=>")
    print(f"Selected file: {files[index]}")

    # pick between napari and neuroglancer
    options = ["neuroglancer", "napari"]
    title = "Select the viewer to use"
    viewer_choice, _ = pick(options, title, indicator="=>")
    is_neuroglancer = viewer_choice == "neuroglancer"

    options = ["yes", "no"]
    title = "Only visualize fibers that pass the validate_fibers criteria?"
    validate_choice, _ = pick(options, title, indicator="=>")
    validate_fibers = validate_choice == "yes"

    # C, Z, Y, X
    im_vol = preprocess_vol(
        imageio.volread(os.path.join(conf.dataset_path, files[index]))
    )
    fiber_seg = h5py.File(
        os.path.join(conf.output_path, files[index].replace(".tif", "-fiber_seg.h5")),
    )
    assert (
        len(fiber_seg.keys()) == 1
    ), "Expected only one key in the fiber segmentation file"
    fiber_seg = fiber_seg[list(fiber_seg.keys())[0]][:]

    final = np.load(
        os.path.join(
            conf.output_path, files[index].replace(".tif", "-final_signals.npz")
        ),
        allow_pickle=True,
    )
    if validate_fibers:
        validate = get_valid(
            final["skel_ids"],
            final["criteria"].item()["lengths"],
            final["criteria"].item()["pca_ratios"],
            final["criteria"].item()["mean_somas"],
            final["criteria"].item()["cell_labels"],
            conf.validate_fibers.thres_length,
            conf.validate_fibers.thres_pca_ratio,
            conf.validate_fibers.thres_mean_soma,
            conf.validate_fibers.one_per_soma,
        )
    else:
        validate = np.ones(len(final["skels"]), dtype=bool)

    final_skels = [skel for skel, valid in zip(final["skels"], validate) if valid]
    final_skel_ids = final["skel_ids"][validate]
    fiber_seg = fastremap.mask_except(fiber_seg, final_skel_ids.tolist())

    kimimaro_skels = [
        points_to_skeleton(skel, id) for skel, id in zip(final_skels, final_skel_ids)
    ]

    cell_seg = h5py.File(
        os.path.join(conf.output_path, files[index].replace(".tif", "-cell_seg.h5")),
    )
    assert (
        len(cell_seg.keys()) == 1
    ), "Expected only one key in the cell segmentation file"
    cell_seg = cell_seg[list(cell_seg.keys())[0]][:]

    assert os.path.exists(conf.output_path), "Output path does not exist"

    layers = {}
    with tempfile.TemporaryDirectory(dir=conf.output_path) as tmpdir:
        if is_neuroglancer:
            cv = to_precomputed(
                fiber_seg,
                os.path.join(tmpdir, "fiber_seg"),
                layer_type="segmentation",
                anisotropy=conf.anisotropy,
                chunk_size=conf.precomputed.chunk_size,
                downsample_factors=conf.precomputed.downsample_factors,
                n_jobs=conf.precomputed.jobs,
                enable_skeletons=True,
            )

            write_skeletons(
                os.path.join(tmpdir, "fiber_seg"),
                kimimaro_skels,
            )
            layers["fiber_seg"] = cv
        else:
            layers["fiber_seg"] = (fiber_seg, "segmentation")

        if is_neuroglancer:
            cv = to_precomputed(
                cell_seg,
                os.path.join(tmpdir, "cell_seg"),
                layer_type="segmentation",
                anisotropy=conf.anisotropy,
                chunk_size=conf.precomputed.chunk_size,
                downsample_factors=conf.precomputed.downsample_factors,
                n_jobs=conf.precomputed.jobs,
            )
            layers["cell_seg"] = cv
        else:
            layers["cell_seg"] = (cell_seg, "segmentation")

        # # for c in [0]:
        for c in range(im_vol.shape[0]):
            if is_neuroglancer:
                output_layer = os.path.join(tmpdir, f"im_{c}")
                cv = to_precomputed(
                    im_vol[c],
                    output_layer,
                    layer_type="image",
                    anisotropy=conf.anisotropy,
                    chunk_size=conf.precomputed.chunk_size,
                    downsample_factors=conf.precomputed.downsample_factors,
                    n_jobs=conf.precomputed.jobs,
                )
                layers[f"im_{c}"] = cv
            else:
                layers[f"im_{c}"] = (im_vol[c], "image")

        if is_neuroglancer:
            viewer = plot_neuroglancer(layers, im_shader=conf.precomputed.im_shader)
        else:
            viewer = plot_napari(
                layers,
                anisotropy=conf.anisotropy,
                to_um=conf.to_um,
                fiber_skels=final_skels,
                fiber_skel_ids=final_skel_ids,
                skel_downsample=conf.napari.skel_downsample,
            )
        input("Press Enter to exit...")
