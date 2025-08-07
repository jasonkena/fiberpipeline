default:
    just --list

fiber_segmentation:
    python -m fiberpipeline.segmentation.fiber_segmentation --config=config.yaml

cell_segmentation:
    python -m fiberpipeline.segmentation.cell_segmentation --config=config.yaml

fiber_skeleton:
    python -m fiberpipeline.processing.skeletonize_fibers --config=config.yaml

extract_signals:
    python -m fiberpipeline.processing.extract_signals --config=config.yaml

normalize_signals:
    python -m fiberpipeline.processing.normalize_signals --config=config.yaml

filter_all:
    python -m fiberpipeline.processing.filter_all --config=config.yaml

stats:
    python -m fiberpipeline.processing.stats --config=config.yaml

plot:
    python -m fiberpipeline.plot.plot --config=config.yaml

plot_signals:
    python -m fiberpipeline.plot.plot_signals --config=config.yaml


everything:
    just fiber_segmentation
    just cell_segmentation
    just fiber_skeleton
    just extract_signals
    just normalize_signals
    just filter_all
    just stats
