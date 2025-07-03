default:
    just --list

fiber_segmentation:
    python fiber_segmentation.py --config=config.yaml

cell_segmentation:
    python cell_segmentation.py --config=config.yaml

plot:
    python plot.py --config=config.yaml
