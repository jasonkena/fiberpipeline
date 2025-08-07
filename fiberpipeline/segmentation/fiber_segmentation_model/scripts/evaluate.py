import sys
from argparse import ArgumentParser
from connectomics.utils.evaluate import adapted_rand
from connectomics.data.utils.data_io import readvol


def evaluate(seg, gt):
    arand, prec, rec = adapted_rand(seg, gt, all_stats=True)
    print(f"Adapted Rand Error: {arand}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--segmentation",
        action="append",
        help="Segmentation file",
        required=True,
    )
    parser.add_argument(
        "--ground_truth",
        action="append",
        help="Ground truth file",
        required=True,
    )
    args = parser.parse_args()
    seg, gt = readvol(args.segmentation[0]), readvol(args.ground_truth[0])
    evaluate(seg, gt)
