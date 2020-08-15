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
Multi-process data loading for AVA.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import atexit
import logging
import numpy as np

from core.config import config as cfg
import datasets.data_input_helper as data_input_helper

import cv2
import datasets.image_processor as imgproc

import random
import copy

logger = logging.getLogger(__name__)
execution_context = None


def create_data_input(
    input_db, expected_data_size, num_processes, num_workers, split,
    batch_size, crop_size=cfg.TRAIN.CROP_SIZE,
):
    # create a global execution context for the dataloader which contains the
    # pool for each thread and each pool has num_processes and a shared data list
    global execution_context

    def init(worker_ids):
        global execution_context

        logging.info('Creating the execution context for '
            'worker_ids: {}, batch size: {}'.format(
                worker_ids,
                batch_size))

        execution_context = data_input_helper._create_execution_context(
            execution_context, _init_pool, worker_ids, expected_data_size,
            num_processes, batch_size)

        atexit.register(_shutdown_pools)

    # in order to get the minibatch, we need some information from the db class
    def get_minibatch_out(
            input_db, worker_id, batch_size, db_indices, crop_size):
        """
        Get minibatch info from AvaDataset and perform the actual
        minibatch loading.
        """
        pools = execution_context.pools
        shared_data_lists = execution_context.shared_data_lists
        curr_pool = pools[worker_id]
        shared_data_list = shared_data_lists[worker_id]

        minibatch_info = input_db.get_minibatch_info(db_indices)

        return _load_and_process_images(
            worker_id, curr_pool, shared_data_list, crop_size,
            minibatch_info, input_db)

    return (init, get_minibatch_out)


def construct_label_array(video_labels):
    """Construction label array."""
    label_arr = np.zeros((cfg.MODEL.NUM_CLASSES, ))

    # AVA label index starts from 1.
    for lbl in video_labels:
        if lbl == -1:
            continue
        assert lbl >= 1 and lbl <= 80
        label_arr[lbl - 1] = 1
    return label_arr.astype(np.int32)


def _load_and_process_images(
    worker_id, curr_pool, shared_data_list, crop_size, minibatch_info, input_db
):
    """Construct a minibatch given minibatch_info."""
    # labels  [box, [l1, l2, l3, ...]]
    
    
    (image_paths, labels, split_list, spatial_shift_positions,
         video_indices, secs, lfb) = minibatch_info[:7]
        
    info_idx = 7
    if cfg.AVABOX.USE_SCENE_LOSS:
        scene_labels = minibatch_info[info_idx]
        info_idx += 1
    
    if cfg.AVABOX.ENABLE_SCENE_FEAT_BANK:
        scene_feat = minibatch_info[info_idx]
        info_idx += 1

        
    if crop_size == cfg.TEST.CROP_SIZE:
        curr_shared_list_id = len(shared_data_list) - 1
    else:
        curr_shared_list_id = 0

    box_list = []
    label_list = []
    for clip_box_labels in labels:
        clip_boxes_arr = []
        clip_label_arr = []
        for box_labels in clip_box_labels:
            clip_boxes_arr.append(box_labels[0])
            clip_label_arr.append(box_labels[1])

        #[(1, 4), (1,4), ...]
        box_list.append(np.array(clip_boxes_arr))

        #[ [ [l1, l2, l3, ...] ...] ...]
        label_list.append(clip_label_arr)

    # Multi-process loading.
    map_results = curr_pool.map_async(
        get_clip_from_source,
        zip(
            [i for i in range(0, len(image_paths))],
            image_paths,
            split_list,
            [crop_size for i in range(0, len(image_paths))],
            spatial_shift_positions,
            box_list,
            [curr_shared_list_id for i in range(0, len(image_paths))],
        )
    )

    results = []
    lbls = []
    boxes = []
    original_boxes = []
    metadata = []

    # Process multi-process loading results.
    for index in map_results.get():
        if index is not None:
            (np_arr, box_arr,
             original_boxes_arr,
             metadata_arr) = shared_data_list[curr_shared_list_id][index]

            if cfg.DATASET == 'avabox' and cfg.AVABOX.CONCAT_GLOBAL_FEAT:
                tmp_np_arr = np.reshape(
                    np_arr, (2, 3, cfg.TRAIN.VIDEO_LENGTH, crop_size, crop_size))
            else:
                tmp_np_arr = np.reshape(
                    np_arr, (3, cfg.TRAIN.VIDEO_LENGTH, crop_size, crop_size))
            
            results.append(tmp_np_arr)

            tmp_box_arr = np.reshape(
                box_arr,
                (cfg.LFB.NUM_LFB_FEAT, 4))
            boxes.append(tmp_box_arr)

            tmp_original_boxes_arr = np.reshape(
                original_boxes_arr,
                (cfg.LFB.NUM_LFB_FEAT, 4))
            original_boxes.append(tmp_original_boxes_arr)

            metadata.append(metadata_arr)

            lbls.append(label_list[index])

    images = data_input_helper.convert_to_batch(results)

    row_idx = 0
    out_boxes = []
    out_original_boxes = []
    out_metadata = []
    out_labels = []
    out_lfb = []

    batch_size = len(lbls)

    for batch in range(batch_size):
        for box_idx in range(len(lbls[batch])):

            row = np.zeros((5,))
            row[0] = batch
            row[1:] = boxes[batch][box_idx]
            out_boxes.append(row)

            row_original = np.zeros((5,))
            row_original[0] = batch
            row_original[1:] = original_boxes[batch][box_idx]

            # video_id, sec, height, width
            row_metadata = np.array([video_indices[batch], secs[batch],
                                     metadata[batch][0], metadata[batch][1]])

            out_original_boxes.append(row_original)
            out_metadata.append(row_metadata)
            
            
            if cfg.JHMDB.ENABLED and cfg.JHMDB.USE_SOFTMAX_LOSS:
                assert len(lbls[batch][box_idx])==1
                out_labels.append(lbls[batch][box_idx][0]-1)
            else:    
                out_labels.append(lbls[batch][box_idx])
                
            if len(lfb) > 0:
                out_lfb.append(lfb[batch])

            row_idx += 1

    out_boxes = np.array(out_boxes).astype(np.float32)
    
        
    if cfg.JHMDB.ENABLED and cfg.JHMDB.USE_SOFTMAX_LOSS:
        out_labels = np.array(out_labels).astype(np.int32)
    else:
        out_labels = np.array([construct_label_array(mult_label)
                           for mult_label in out_labels])
        
    out_original_boxes = np.array(out_original_boxes).astype(np.float32)
    out_metadata = np.array(out_metadata).astype(np.float32)
    out_lfb = np.array(out_lfb).astype(np.float32)
    
    out_list = [images, out_labels, out_boxes,
                out_original_boxes, out_metadata,
                out_lfb]
    
    if cfg.AVABOX.USE_SCENE_LOSS:
        scene_labels = np.array([construct_label_array(mult_label)
                           for mult_label in scene_labels])
        out_list.append(scene_labels)
        
    if cfg.AVABOX.ENABLE_SCENE_FEAT_BANK:
        scene_feat = np.array(scene_feat).astype(np.float32)
        out_list.append(scene_feat)
    
    return out_list 


def scale_box(box, scale_list):
    """
    box: (4,)  [w1, h1, w2, h2]
    """
    w1, h1, w2, h2 = box
    center_w = (w1 + w2) / 2
    center_h = (h1 + h2) / 2
    
    half_w = center_w - w1
    half_h = center_h - h1
    
    scale = random.uniform(scale_list[0], scale_list[1])
    
    half_w = scale * half_w
    half_h = scale * half_h
    
    new_box = np.array([center_w-half_w, center_h-half_h,
           center_w+half_w, center_h+half_h])
    
    return new_box


def scale_box_xy(box, scale_list):
    """
    box: (4,)  [w1, h1, w2, h2]
    """
    w1, h1, w2, h2 = box
    center_w = (w1 + w2) / 2
    center_h = (h1 + h2) / 2
    
    half_w = center_w - w1
    half_h = center_h - h1
    
    scale = random.uniform(scale_list[0], scale_list[1])
    half_w1 = scale * half_w
    
    scale = random.uniform(scale_list[0], scale_list[1])
    half_w2 = scale * half_w
    
    scale = random.uniform(scale_list[0], scale_list[1])
    half_h1 = scale * half_h
    
    scale = random.uniform(scale_list[0], scale_list[1])
    half_h2 = scale * half_h
    
    new_box = np.array([center_w-half_w1, center_h-half_h1,
           center_w+half_w2, center_h+half_h2])
    
    return new_box
    
    

def get_clip_from_source(args):
    (index, image_paths, split, crop_size,
     spatial_shift_pos, boxes, list_id) = args
    """Load images/data from disk and pre-process data."""
    """
    boxes: (1, 4)
    """

    try:
        if cfg.AVABOX.CONCAT_GLOBAL_FEAT and (cfg.AVABOX.USE_LONG_TERM_FEAT or cfg.JHMDB.USE_FLOW):
            imgs = data_input_helper.retry_load_images(image_paths[0],
                                                   cfg.IMG_LOAD_RETRY)
            imgs_global = data_input_helper.retry_load_images(image_paths[1],
                                                   cfg.IMG_LOAD_RETRY)
        else:
            imgs = data_input_helper.retry_load_images(image_paths,
                                                   cfg.IMG_LOAD_RETRY)
            if cfg.AVABOX.CONCAT_GLOBAL_FEAT:
                imgs_global = copy.deepcopy(imgs)
                
        
        # Score is not used.
        boxes = boxes[:, :4].copy()
        original_boxes = boxes.copy()

        height, width, _ = imgs[0].shape
        
        
        assert boxes.shape[0] == 1
        box = boxes.copy()[0]
        
        # scale the box
        if split == 1: # "train"
            if cfg.AVABOX.SCALE_XY:
                box = scale_box_xy(box, cfg.AVABOX.TRAIN_BOX_SCALES)
            else:
                box = scale_box(box, cfg.AVABOX.TRAIN_BOX_SCALES)
        else:
            assert cfg.AVABOX.TEST_BOX_SCALES[0] == cfg.AVABOX.TEST_BOX_SCALES[1]
            box = scale_box(box, cfg.AVABOX.TEST_BOX_SCALES)
            
        
        boxes = np.array(box.copy()).reshape(1,4)
        
        # [0, 1] -> [0, H,W]
        box[[0, 2]] *= width
        box[[1, 3]] *= height
        box = imgproc.clip_box_to_image(box, height, width)
        box = box.astype(np.int)
        
        if cfg.AVABOX.ERASE_BACKGROUND_EXP:
            for i in range(len(imgs)):
                imgs[i][:box[1],:,:]=0
                imgs[i][box[3]+1:,:,:]=0

                imgs[i][:,:box[0],:]=0
                imgs[i][:,box[2]+1:,:]=0
                
            imgs, boxes = data_input_helper.images_and_boxes_preprocessing(
                imgs, split, crop_size, spatial_shift_pos, boxes=boxes)
            
        else:
            flip_flag = ''
            if cfg.AVABOX.CONCAT_GLOBAL_FEAT:
                if split == 1:
                    if np.random.uniform() < 0.5:
                        flip_flag = 'y'
                    else:
                        flip_flag = 'n'

                imgs_global, boxes = data_input_helper.images_and_boxes_preprocessing(
                            imgs_global, split, crop_size, spatial_shift_pos, boxes=boxes,
                            flip_flag = flip_flag)


            # crop box from imgs 
            # Now the image is in HWC, BGR format
            imgs = [ image[ box[1]:box[3], box[0]:box[2], :]
                            for image in imgs]
            height, width, _ = imgs[0].shape

            imgs = data_input_helper.images_preprocessing_avabox(
                imgs, split, crop_size, spatial_shift_pos,
                flip_flag = flip_flag)

            if cfg.AVABOX.CONCAT_GLOBAL_FEAT:
                # 2, 3, T, H, W
                imgs = np.stack([imgs, imgs_global])
        
        
        
        
        
        # loading images
        np_arr = shared_data_list[list_id][index][0]
        np_arr = np.reshape(np_arr, imgs.shape)

        np_arr[:] = imgs

        # loading boxes
        box_arr = shared_data_list[list_id][index][1]

        box_arr = np.reshape(box_arr, (
            cfg.LFB.NUM_LFB_FEAT, 4))
        box_arr[:boxes.shape[0]] = boxes


        # loading original boxes
        original_box_arr = shared_data_list[list_id][index][2]

        original_box_arr = np.reshape(original_box_arr, (
            cfg.LFB.NUM_LFB_FEAT, 4))
        original_box_arr[:original_boxes.shape[0]] = original_boxes


        # loading metadata
        metadata_arr = shared_data_list[list_id][index][3]
        metadata_arr[0] = height
        metadata_arr[1] = width
        
        

    except Exception as e:
        logger.error('get_image_from_source failed: '
                     '(index, image_path, split): {} {} {}'.format
                     (index, image_paths, split))
        logger.info(e)
        return None
    return index


def _init_pool(data_list):
    """
    Each pool process calls this initializer.
    Load the array to be populated into that process's global namespace.
    """
    global shared_data_list
    shared_data_list = data_list


def _shutdown_pools():
    data_input_helper._shutdown_pools(execution_context.pools)
