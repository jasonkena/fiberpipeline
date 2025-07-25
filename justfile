default:
    just --list

fiber_segmentation:
    python fiber_segmentation.py --config=config.yaml

cell_segmentation:
    python cell_segmentation.py --config=config.yaml

process_segmentation:
    python process_segmentation.py --config=config.yaml

fiber_skeleton:
    python skeletonize_fibers.py --config=config.yaml

extract_signals:
    python extract_signals.py --config=config.yaml

normalize_signals:
    python normalize_signals.py --config=config.yaml

filter_all:
    python filter_all.py --config=config.yaml

plot:
    python plot.py --config=config.yaml

plot_signals:
    python plot_signals.py --config=config.yaml

everything:
    just fiber_segmentation
    just cell_segmentation
    just fiber_skeleton
    just extract_signals
    just normalize_signals
    just filter_all
