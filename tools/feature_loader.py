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

"""Tool to construct, write or load an LFB."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace
import logging
import numpy as np
import os
import pickle
import sys

from core.config import config as cfg
from models import model_builder_video
from utils.timer import Timer
import utils.checkpoints as checkpoints
import utils.misc as misc


FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_features(feature_name):
    """Get features from all GPUs given name."""
    features = []
    for idx in range(cfg.ROOT_GPU_ID, cfg.ROOT_GPU_ID + cfg.NUM_GPUS):
        features.append(workspace.FetchBlob(
            'gpu_{}/{}'.format(idx, feature_name)))
    return features


# def construct_frame_level_lfb(all_features, all_metadata):
#     """Construct an frame-level LFB (e.g., for EPIC-Kitchens and Chrades)."""
#     lfb = {}

#     global_idx = 0
#     for iter_features in all_features:
#         for gpu_features in iter_features:
#             batch_size = gpu_features.shape[0]

#             for i in range(batch_size):
#                 if global_idx >= len(all_metadata):
#                     break
#                 if cfg.DATASET == 'epic':
#                     _, video_id, frame_id, _, _, _ = all_metadata[global_idx]
#                 elif cfg.DATASET == 'charades':
#                     video_id, frame_id = all_metadata[global_idx]
#                 global_idx += 1

#                 if video_id not in lfb:
#                     lfb[video_id] = {}

#                 lfb[video_id][frame_id] = np.squeeze(gpu_features[i])

#     logger.info('LFB constructed')
#     logger.info('\t%d frames in %d videos (%.03f frames / video).' % (
#         global_idx, len(lfb), float(global_idx) / len(lfb)))

#     return lfb


def construct_ava_lfb(all_features, all_metadata,
                      all_labels, all_proposals, all_original_boxes):
    """Construct an LFB for AVA."""
    total_sec = 0
    num_boxes = 0
    lfb = {}
    
    for iter_index in range(len(all_metadata)):
        for gpu_index in range(len(all_metadata[iter_index])):
            
            gpu_metadata = all_metadata[iter_index][gpu_index]
            gpu_labels = all_labels[iter_index][gpu_index]
            gpu_proposals = all_proposals[iter_index][gpu_index]
            gpu_original_boxes = all_original_boxes[iter_index][gpu_index]
            
            num_rois = gpu_metadata.shape[0]
            
            for i in range(num_rois):
                video_id, sec, height, width = gpu_metadata[i].tolist()
                video_id = int(np.round(video_id))
                sec = int(np.round(sec))

                if video_id not in lfb:
                    lfb[video_id] = {}

                if sec not in lfb[video_id]:
                    lfb[video_id][sec] = {}
                    total_sec += 1
                    
                    lfb[video_id][sec]['labels'] = []
                    lfb[video_id][sec]['proposals'] = []
                    lfb[video_id][sec]['original_boxes'] = []
                    for feat_name in cfg.FEATURE_MAP_LOADER.NAME_LIST:
                        lfb[video_id][sec][feat_name] = []
                
                for feat_name in cfg.FEATURE_MAP_LOADER.NAME_LIST:
                    gpu_features=all_features[feat_name][iter_index][gpu_index]
                    lfb[video_id][sec][feat_name].append(np.squeeze(gpu_features[i]))
                
                lfb[video_id][sec]['labels'].append(gpu_labels[i])
                lfb[video_id][sec]['proposals'].append(gpu_proposals[i])
                lfb[video_id][sec]['original_boxes'].append(gpu_original_boxes[i])
                
                num_boxes += 1

                
    logger.info('LFB constructed')
    logger.info('\t%d seconds in %d videos (%.03f secs / videos).' % (
        total_sec, len(lfb), float(total_sec) / len(lfb)))

    logger.info('\t%d boxes in total (%.03f boxes / sec).' % (
        num_boxes, float(num_boxes) / total_sec))
    return lfb




def write_lfb(lfb, is_train):
    """Write LFB to a pickle file."""
    out_lfb_filename = os.path.join(
        cfg.FEATURE_MAP_LOADER.OUT_DIR,
        'train_features.pkl' if is_train else 'val_features.pkl')
    with open(out_lfb_filename, 'wb') as f:
        pickle.dump(lfb, f, pickle.HIGHEST_PROTOCOL)
    logger.info('Inferred features saved as %s.' % out_lfb_filename)


def construct_lfb(features, metadata, 
                  labels, proposals, original_boxes, 
                  input_db, is_train):
    if cfg.DATASET == "ava":
        lfb = construct_ava_lfb(features, metadata,
                                labels, proposals, original_boxes)
#         assert len(lfb) == (cfg.TRAIN.DATASET_SIZE if is_train
#                             else cfg.TEST.DATASET_SIZE)

#     elif cfg.DATASET == 'charades':
#         lfb = construct_frame_level_lfb(features, input_db._lfb_frames)
#         assert len(lfb) == (cfg.TRAIN.DATASET_SIZE if is_train
#                             else cfg.TEST.DATASET_SIZE)

#     elif cfg.DATASET == 'epic':
#         lfb = construct_frame_level_lfb(features, input_db._annotations)
#         assert len(lfb) == len(input_db._image_paths)

    else:
        raise Exception("Dataset {} not recognized.".format(cfg.DATASET))
    return lfb


def load_feature_map(params_file, is_train):
    assert params_file, 'FEATURE_MAP_LOADER.MODEL_PARAMS_FILE is not specified.'
    assert cfg.FEATURE_MAP_LOADER.OUT_DIR, 'FEATURE_MAP_LOADER.OUT_DIR is not specified.'
    logger.info('Inferring feature map from %s' % params_file)
    
    cfg.FEATURE_MAP_LOADER.ENALBE = True

    cfg.GET_TRAIN_LFB = is_train

    timer = Timer()

    test_model = model_builder_video.ModelBuilder(
        train=False,
        use_cudnn=True,
        cudnn_exhaustive_search=True,
        split=cfg.TEST.DATA_TYPE,
    )

    suffix = 'infer_{}'.format('train' if is_train else 'test')
    
    
    if cfg.LFB.ENABLED:
        lfb_path = os.path.join(cfg.LFB.LOAD_LFB_PATH,
                                'train_lfb.pkl' if is_train
                                else 'val_lfb.pkl')
        logger.info('Loading LFB from %s' % lfb_path)
        with open(lfb_path, 'r') as f:
            lfb = pickle.load(f)
            
        test_model.build_model(
            lfb=lfb,
            suffix=suffix,
            shift=1,
        )
        
    else:
        test_model.build_model(
            lfb=None,
            suffix=suffix,
            shift=1,
        )

    if cfg.PROF_DAG:
        test_model.net.Proto().type = 'prof_dag'
    else:
        test_model.net.Proto().type = 'dag'

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net)

    total_test_net_iters = misc.get_total_test_iters(test_model)

    test_model.start_data_loader()

    checkpoints.load_model_from_params_file_for_test(
        test_model, params_file)

    all_features = {}
    for feat_name in cfg.FEATURE_MAP_LOADER.NAME_LIST:
        all_features[feat_name] = []
    
    all_metadata = []
    
    all_labels = []
    all_proposals = []
    all_original_boxes = []
    
    if cfg.FEATURE_MAP_LOADER.TEST_ITERS > 0:
        total_test_net_iters = cfg.FEATURE_MAP_LOADER.TEST_ITERS

    for test_iter in range(total_test_net_iters):

        timer.tic()
        workspace.RunNet(test_model.net.Proto().name)
        timer.toc()

        if test_iter == 0:
            misc.print_net(test_model)
            os.system('nvidia-smi')
        if test_iter % 10 == 0:
            logger.info("Iter {}/{} Time: {}".format(
                test_iter, total_test_net_iters, timer.diff))

        if cfg.DATASET == "ava":
            for feat_name in cfg.FEATURE_MAP_LOADER.NAME_LIST:
                all_features[feat_name].append(get_features(feat_name))
            
            all_metadata.append(get_features('metadata{}'.format(suffix)))
            
            all_labels.append(get_features('labels{}'.format(suffix)))
            all_proposals.append(get_features('proposals{}'.format(suffix)))
            all_original_boxes.append(get_features('original_boxes{}'.format(suffix)))            
            
#         elif cfg.DATASET in ['charades', 'epic']:
#             all_features.append(get_features('pool5'))
        else:
            raise Exception("Dataset {} not recognized.".format(cfg.DATASET))

    
    
    lfb = construct_lfb(
        all_features, all_metadata, 
        all_labels, all_proposals, all_original_boxes, 
        test_model.input_db, is_train)
    
    write_lfb(lfb, is_train)
    
    logger.info("Shutting down data loader...")
    test_model.shutdown_data_loader()

    workspace.ResetWorkspace()
    logger.info("Done ResetWorkspace...")

    cfg.GET_TRAIN_LFB = False

