import sys
import os

PROJECT_ROOT=os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..'
        )
    )
sys.path.append(PROJECT_ROOT)

REPO_ROOT=os.path.abspath(
    os.path.join(
        PROJECT_ROOT, '..', '..'
    )
)
sys.path.append(REPO_ROOT)

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)

import numpy as np

from dcvt.common.utils.config import read_config
from dcvt.detection.utils.anchors import YoloAnchorsGenerator

from dcvt.detection.train import load_datasets


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Construct the anchors by config',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str)
    args = vars(parser.parse_args())
    logger.debug(f'Input args: {args}')
    return args


if __name__ == '__main__':
    args = get_args()
    config = read_config(args['config'])

    datasets = load_datasets(config)
    anchors_dataset = datasets['preprocessed'][0]
    target_dims = []

    print(f'Collecting bboxes from {len(anchors_dataset)} samples')
    for i in range(len(anchors_dataset)):
        img, ann = anchors_dataset[i]
        bboxes = ann['bboxes']
        for bbox in bboxes:
            if np.sum(bbox) == 0:
                break

            target_dims.append(
                bbox[2:4]
            )

    ANCHORS_COUNT = len(config.model.anchor_masks) * len(config.model.anchor_masks[0])
    anchors_gen = YoloAnchorsGenerator(ANCHORS_COUNT)

    centroids = anchors_gen.generate(target_dims)
    centroids_str = anchors_gen.centroids_as_string(centroids)

    logger.debug(f'Anchors: {centroids_str}')
