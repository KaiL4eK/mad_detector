import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".."))
sys.path.append(REPO_ROOT)

import logging

from dcvt.detection.train import train
from dcvt.common.utils.config import read_multiple_configs
from dcvt.common.utils.args import get_args, parse_args_params
from dcvt.common.utils.common import merge_update_dicts
from dcvt.common.utils import remote
from dcvt.common.utils.logger import sample_init_logger


if __name__ == "__main__":
    logger = sample_init_logger(logging.DEBUG)
    args = get_args()
    config = read_multiple_configs(args["config"])

    # Get params and update config from args
    params_dict = parse_args_params(args["params"])
    config = merge_update_dicts(config, params_dict)

    logger.debug(f"Config: {config}")

    logger.debug(f"Initialize remote")
    remote.init()

    train(config=config, dry_run=args["dry_run"])
