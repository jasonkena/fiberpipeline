import glob
import os
import numpy as np
import neuroglancer
import imageio
import h5py
from pick import pick
from nicety.conf import get_conf, DotDict
from utils import preprocess_vol
from precomputed import to_precomputed, write_skeletons
import tempfile
from typing import Dict
from cloudvolume import CloudVolume
import socketserver
import threading
from urllib.parse import urlparse
import napari
from napari import Viewer


def get_free_port(ip: str) -> int:
    # https://stackoverflow.com/a/61685162/10702372
    with socketserver.TCPServer((ip, 0), None) as s:
        return s.server_address[1]


def start_server(cv: CloudVolume, port):
    cv.viewer(port=port)


def serve_cloudvolume_layers(ip: str, layers: Dict[str, CloudVolume]) -> DotDict:
    """
    Start multiple CloudVolume.viewer() instances

    """
    threads = {}
    for name, cv in layers.items():
        port = get_free_port(ip)
        thread = threading.Thread(target=start_server, args=(cv, port))
        print(f"Starting server for {name} on port {port}")
        thread.start()
        threads[name] = {
            "thread": thread,
            "port": port,
            "cv": cv,
        }

    return DotDict(threads)


def plot_neuroglancer(
    layers: Dict[str, CloudVolume],
):
    ip = "localhost"
    neuroglancer.set_server_bind_address(bind_address=ip)
    viewer = neuroglancer.Viewer()

    threads = serve_cloudvolume_layers(ip, layers)
    ports = [threads[name].port for name in layers.keys()]

    with viewer.txn() as s:
        for name in layers.keys():
            assert layers[name].layer_type in ["image", "segmentation"]

            if layers[name].layer_type == "image":
                s.layers.append(
                    name=name,
                    layer=neuroglancer.ImageLayer(
                        source=f"precomputed://http://{ip}:{threads[name].port}",
                    ),
                )
            else:
                cv = layers[name]
                s.layers.append(
                    name=name,
                    layer=neuroglancer.SegmentationLayer(
                        source=f"precomputed://http://{ip}:{threads[name].port}",
                    ),
                    # tell NG which segments exist
                    segments=sorted(map(int, cv.image.unique(cv.bounds, cv.mip))),
                )

    print(viewer)
    ports.append(urlparse(str(viewer)).port)
    ports = " ".join(map(str, ports))
    print(f"Ports: {ports}")

    return viewer


def plot_napari(layers: Dict[str, CloudVolume], mip):
    """
    Plot layers in napari.
    """
    viewer = Viewer()
    for name, cv in layers.items():
        assert cv.layer_type in ["image", "segmentation"]

        _cv = CloudVolume(cv.layerpath, mip=mip)
        vol = _cv[:].squeeze(-1)  # remove the last dimension
        scale = _cv.scale["resolution"]
        func = viewer.add_image if cv.layer_type == "image" else viewer.add_labels

        func(vol, name=name, scale=scale)
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

    fiber_skels = np.load(
        os.path.join(conf.output_path, files[index].replace(".tif", "-fiber_skel.npz")),
        allow_pickle=True,
    )["skels"].tolist()
    # fiber_skels = np.load(
    #     os.path.join(conf.output_path, files[index].replace(".tif", "-filtered_fiber_skel.npz")),
    #     allow_pickle=True,
    # )["skels"].tolist()

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
            fiber_skels,
        )
        layers["fiber_seg"] = cv

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

        for c in range(im_vol.shape[0]):
            output_layer = os.path.join(tmpdir, f"im_{c}")

            cv = to_precomputed(
                im_vol[c],
                output_layer,
                layer_type="image",
                anisotropy=conf.anisotropy,
                chunk_size=conf.precomputed.chunk_size,
                n_jobs=conf.precomputed.jobs,
            )
            layers[f"im_{c}"] = cv

        if is_neuroglancer:
            viewer = plot_neuroglancer(layers)
        else:
            viewer = plot_napari(layers, mip=conf.napari.mip)
        input("Press Enter to exit...")
