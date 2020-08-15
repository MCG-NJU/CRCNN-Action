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

"""
Dataset class for AVA. It stores dataset information such as frame paths,
annotations, etc. It also provides a function for generating "minibatch_info",
which consists of the necessary information for constructing a minibatch
(e.g. frame paths, LFB, labels, etc. ). The actual data loading from disk and
construction of data blobs will be performed by ava_data_input.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import config as cfg
import cv2
import logging
import numpy as np
import os
import random

import copy

import datasets.dataset_helper as dataset_helper


cv2.ocl.setUseOpenCL(False)
logger = logging.getLogger(__name__)


def load_boxes_and_labels(filenames, is_train, detect_thresh, full_eval):
    """Loading boxes and labels from csv files."""
    ret = {}
    count = 0
    unique_box_count = 0
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                assert len(row) == 7 or len(row) == 8
                video_name, frame_sec = row[0], int(row[1])

                # We "define" the subset of AVA validation set to be the
                # frames where frame_sec % 4 == 0.
                if (not is_train and not full_eval and frame_sec % 4 != 0):
                    continue

                box_key = ','.join(row[2:6])
                box = map(float, row[2:6])
                label = -1 if row[6] == '' else int(row[6])

                if 'predicted' in filename:
                    # When we use predicted boxes to train/eval, we have scores.
                    score = float(row[7])
                    if score < detect_thresh:
                        continue

                if video_name not in ret:
                    ret[video_name] = {}
                    
                if frame_sec not in ret[video_name]:
                    ret[video_name][frame_sec] = {}

                if box_key not in ret[video_name][frame_sec]:
                    if is_train: 
                        if cfg.TRAIN.MAX_BOX_NUM is not None and len(ret[video_name][frame_sec]) >= cfg.TRAIN.MAX_BOX_NUM:
                            continue

                    ret[video_name][frame_sec][box_key] = [box, []]
                    unique_box_count += 1

                ret[video_name][frame_sec][box_key][1].append(label)
                if label != -1:
                    count += 1

    for video_name in ret.keys():
        for frame_sec in ret[video_name].keys():
            ret[video_name][frame_sec] = ret[video_name][frame_sec].values()

    logger.info('Finished loading annotations from')
    for filename in filenames:
        logger.info("  %s" % filename)
    logger.info("Number of unique boxes: %d" % unique_box_count)
    logger.info("Number of annotations: %d" % count)
    return ret


def get_keyframe_indices(boxes_and_labels):
    """
    Getting keyframe indices that will be used for training and testing.
    For training, we only need to train on frames with boxes/labels.
    For testing, we also only need to predict for frames with predicted boxes.
    """
    keyframe_indices = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        for sec in boxes_and_labels[video_idx].keys():

            if len(boxes_and_labels[video_idx][sec]) > 0:
                keyframe_indices.append((video_idx, sec))
                count += 1
    logger.info("%d keyframes used." % count)
    return keyframe_indices


def get_box_indices(boxes_and_labels):
    """
    Getting box indices that will be used for training and testing.
    For training, we only need to train on frames with boxes/labels.
    For testing, we also only need to predict for frames with predicted boxes.
    """
    box_indices = []
    
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        for sec in boxes_and_labels[video_idx].keys():

            for box_idx in range(len(boxes_and_labels[video_idx][sec])):
                box_indices.append((video_idx, sec, box_idx))
                count += 1
                
    logger.info("%d boxes used." % count)

    return box_indices



def get_num_boxes_used(keyframe_indices, boxes_and_labels):
    """Get total number of boxes used."""
    count = 0
    for video_idx, sec in keyframe_indices:
        count += len(boxes_and_labels[video_idx][sec])
    return count


class JHMDBDataset():

    def __init__(self, split, lfb_infer_only, shift=None, lfb=None, suffix=''):

        self.blobnames = [blobname + suffix for blobname in [
            'data',
            'labels',
            'proposals',
            'original_boxes',
            'metadata',
            'lfb',
        ]]

        self._split = split
        self._lfb_infer_only = lfb_infer_only
        self._shift = shift

        if lfb_infer_only:
            self._lfb_enabled = False
            # For LFB inference, we use all keyframes, so _full_eval is true.
            self._full_eval = True
            self._detect_thresh = cfg.AVA.LFB_DETECTION_SCORE_THRESH
        else:
            self._lfb_enabled = cfg.LFB.ENABLED
            self._full_eval = cfg.AVA.FULL_EVAL
            self._detect_thresh = cfg.AVA.DETECTION_SCORE_THRESH

        # This is used for selecting data augmentation.
        self._split_num = int(split == 'train' and not lfb_infer_only)

        self._get_data()

        if self._split == 'train':
            self._sample_rate = cfg.TRAIN.SAMPLE_RATE
            self._video_length = cfg.TRAIN.VIDEO_LENGTH
            self._batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            self._sample_rate = cfg.TEST.SAMPLE_RATE
            self._video_length = cfg.TEST.VIDEO_LENGTH
            self._batch_size = cfg.TEST.BATCH_SIZE
            
            
        self._seq_len = self._video_length * self._sample_rate

        if self._lfb_enabled:
            self._lfb = lfb
            assert len(self._image_paths) == len(self._lfb), \
                'num videos %d != num videos in LFB %d' % (
                    len(self._image_paths), len(self._lfb))

    def get_db_size(self):
        return len(self._box_indices)

    def get_minibatch_info(self, indices):
        """
        Given iteration indices, return the necessarry information for
        constructing a minibatch. This will later be used in ava_data_input.py
        to actually load the data and constructing blobs.
        """
        half_len = self._seq_len // 2
        labels = []
        image_paths = []
        secs = []
        video_indices = []
        lfb = []

        if not isinstance(indices, list):
            indices = indices.tolist()

        while len(indices) < self._batch_size // cfg.NUM_GPUS:
            indices.append(indices[0])

        for idx in indices:
            if self._split == 'train':
                rand_idx = np.random.randint(0, len(self._box_indices))
                video_idx, sec, box_idx = self._box_indices[rand_idx]
            else:
                video_idx, sec, box_idx = self._box_indices[idx]
                

            center_idx = sec - 1 
                
            seq = dataset_helper.get_sequence(
                center_idx, half_len, self._sample_rate,
                num_frames=len(self._image_paths[video_idx]))
            
            img_list = [self._image_paths[video_idx][frame] for frame in seq]

            if cfg.AVABOX.CONCAT_GLOBAL_FEAT and cfg.JHMDB.USE_FLOW:
                
                flow_img_list = []
                for path in img_list:
                    flow_path = path.replace('Frames', 'Flow')
                    flow_path = flow_path.replace('png', 'jpg')
                    flow_path = flow_path.encode("ascii")
                    flow_img_list.append(flow_path)
                
                image_paths.append([img_list, flow_img_list])
            else:
                image_paths.append(img_list)

            clip_label_list = [ self._boxes_and_labels[video_idx][sec][box_idx] ]
            assert len(clip_label_list) > 0

            secs.append(sec)
            video_indices.append(video_idx)

            labels.append(clip_label_list)

            if self._lfb_enabled:
                lfb.append(sample_lfb(self._lfb[video_idx], sec, self._split=='train'))
                

        split_list = [self._split_num] * len(indices)

        spatial_shift_positions = [CENTER_CROP_INDEX if self._shift is None
                                   else self._shift] * len(indices)

        return (image_paths, labels, split_list, spatial_shift_positions,
                video_indices, secs, lfb)

    def _get_data(self):
        """Load frame paths and annotations. """
        
        if cfg.JHMDB.SPLIT_NUM == 0:
            if self._split == 'train':
                list_filenames = [os.path.join(cfg.JHMDB.FRAME_LIST_DIR, 
                                              'frame_list_train.csv')]
            else:
                list_filenames = [os.path.join(cfg.JHMDB.FRAME_LIST_DIR, 
                                              'frame_list_test.csv')]
        else:
            if self._split == 'train':
                list_filenames = [os.path.join(cfg.JHMDB.FRAME_LIST_DIR, 
                                              'frame_list_train_split%d.csv'%cfg.JHMDB.SPLIT_NUM)]
            else:
                list_filenames = [os.path.join(cfg.JHMDB.FRAME_LIST_DIR, 
                                              'frame_list_test_split%d.csv'%cfg.JHMDB.SPLIT_NUM)]
                
        
        (self._image_paths, _,
         self._video_idx_to_name, _) = dataset_helper.load_image_lists(
            list_filenames)

        if cfg.JHMDB.SPLIT_NUM == 0:
            if self._split == 'train':
                ann_filenames = [os.path.join(cfg.JHMDB.ANNOTATION_DIR, 
                                              'train.csv')]

                if cfg.JHMDB.USE_PRED_BOX_TRAIN:
                    ann_filenames.append(os.path.join(cfg.JHMDB.ANNOTATION_DIR, 
                                              'predicted_train.csv'))

            else:
                if cfg.JHMDB.USE_GT_BOX_TEST:
                    ann_filenames = [os.path.join(cfg.JHMDB.ANNOTATION_DIR, 
                                              'test.csv')]
                else:
                    ann_filenames = [os.path.join(cfg.JHMDB.ANNOTATION_DIR, 
                                              'max_predicted_test.csv')]

            cfg.JHMDB.EVAL_GT_FILENAME = 'test.csv'
        else:
            if self._split == 'train':
                ann_filenames = [os.path.join(cfg.JHMDB.ANNOTATION_DIR, 
                                              'train_split%d.csv'%cfg.JHMDB.SPLIT_NUM)]

                if cfg.JHMDB.USE_PRED_BOX_TRAIN:
                    ann_filenames.append(os.path.join(cfg.JHMDB.ANNOTATION_DIR, 
                                              'ft_ep12_res50_pred/predicted_train_split%d.csv'%cfg.JHMDB.SPLIT_NUM))

            else:
                if cfg.JHMDB.USE_GT_BOX_TEST:
                    ann_filenames = [os.path.join(cfg.JHMDB.ANNOTATION_DIR, 
                                              'test_split%d.csv'%cfg.JHMDB.SPLIT_NUM)]
                else:
                    ann_filenames = [os.path.join(cfg.JHMDB.ANNOTATION_DIR, 
                                              'ft_ep12_res50_pred/predicted_test_split%d.csv'%cfg.JHMDB.SPLIT_NUM)]

            cfg.JHMDB.EVAL_GT_FILENAME = 'test_split%d.csv'%cfg.JHMDB.SPLIT_NUM
            
                        
        self._boxes_and_labels = load_boxes_and_labels(
            ann_filenames,
            is_train=(self._split == 'train'),
            detect_thresh=self._detect_thresh,
            full_eval=self._full_eval)

        
        self.new_boxes_and_labels = []
        for i in range(len(self._image_paths)):
            vname = self._video_idx_to_name[i]
            
            if vname in self._boxes_and_labels:
                self.new_boxes_and_labels.append(self._boxes_and_labels[vname])
            else:
                self.new_boxes_and_labels.append({})
                
        self._boxes_and_labels = self.new_boxes_and_labels

        
        self._keyframe_indices = get_keyframe_indices(self._boxes_and_labels)
        
        self._box_indices = get_box_indices(self._boxes_and_labels)

        
        self._num_boxes_used = get_num_boxes_used(
            self._keyframe_indices, self._boxes_and_labels)

        self.print_summary()

    def print_summary(self):
        logger.info("=== JHMDB dataset summary ===")
        logger.info('Split: {}'.format(self._split))
        logger.info("Use LFB? {}".format(self._lfb_enabled))
        logger.info("Detection threshold: {}".format(self._detect_thresh))
        if self._split != 'train':
            logger.info("Full evaluation? {}".format(self._full_eval))
        logger.info('Spatial shift position: {}'.format(self._shift))
        logger.info('Number of videos: {}'.format(len(self._image_paths)))
        total_frames = sum(len(video_img_paths)
                           for video_img_paths in self._image_paths)
        logger.info('Number of frames: {}'.format(total_frames))
        logger.info('Number of dataset: {}'.format(self.get_db_size()))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))
        
        logger.info('JHMDB split num: %d.'%cfg.JHMDB.SPLIT_NUM)

