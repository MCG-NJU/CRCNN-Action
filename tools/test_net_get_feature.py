# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Train a video model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace
import argparse
import logging
import os
import sys
from tensorboardX import SummaryWriter

from core.config import assert_and_infer_cfg
from core.config import cfg_from_file
from core.config import cfg_from_list
from core.config import config as cfg
from core.config import print_cfg
from feature_loader import load_feature_map
from models import model_builder_video
from test_net import test_net
from utils.timer import Timer
import utils.bn_helper as bn_helper
import utils.c2 as c2_utils
import utils.checkpoints as checkpoints
import utils.metrics as metrics
import utils.misc as misc

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)




def test(opts):
    """Get feature map from model."""

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logging.getLogger(__name__)

    # Generate seed.
    misc.generate_random_seed(opts)

    # Setting training-time-specific configurations.
    cfg.AVA.FULL_EVAL = cfg.AVA.FULL_EVAL_DURING_TRAINING
    cfg.AVA.DETECTION_SCORE_THRESH = cfg.AVA.DETECTION_SCORE_THRESH_TRAIN
    cfg.CHARADES.NUM_TEST_CLIPS = cfg.CHARADES.NUM_TEST_CLIPS_DURING_TRAINING


    cfg.FEATURE_MAP_LOADER.OUT_DIR = cfg.CHECKPOINT.DIR
    load_feature_map(cfg.FEATURE_MAP_LOADER.MODEL_PARAMS_FILE, is_train=False)
#     load_feature_map(cfg.FEATURE_MAP_LOADER.MODEL_PARAMS_FILE, is_train=True)


def main():
    c2_utils.import_detectron_ops()
    parser = argparse.ArgumentParser(description='Classification model testing')
    parser.add_argument('--config_file', type=str, default=None, required=True,
                        help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
                        default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)

    assert_and_infer_cfg()
    print_cfg()

    test(args)


if __name__ == '__main__':
    main()
