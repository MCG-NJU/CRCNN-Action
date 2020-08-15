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

"""3D ResNet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from core.config import config as cfg
from utils.misc import get_batch_size
import models.head_helper as head_helper
import models.resnet_helper as resnet_helper

logger = logging.getLogger(__name__)

import numpy as np


BLOCK_CONFIG = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
}


def obtain_arc(arc_type):
    """
    Architecture definition.
    This function defines the temporal kernel radius and temporal strides for
    each layer in a ResNet.
    For example, use_temp_convs = 1 stands for a temporal kernel size of 3.
    In ResNet50, it has (3, 4, 6, 3) blocks in conv2, 3, 4, 5.
    so the lengths of the corresponding lists are (3, 4, 6, 3).
    """
    pool_stride = 1

    # C2D, ResNet50.
    if arc_type == 1:
        use_temp_convs_1 = [0]
        temp_strides_1 = [1]
        use_temp_convs_2 = [0, 0, 0]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [0, 0, 0, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [0, ] * 6
        temp_strides_4 = [1, ] * 6
        use_temp_convs_5 = [0, 0, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)

    # I3D, ResNet50.
    if arc_type == 2:
        use_temp_convs_1 = [2]
        temp_strides_1 = [1]
        use_temp_convs_2 = [1, 1, 1]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [1, 0, 1, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [1, 0, 1, 0, 1, 0]
        temp_strides_4 = [1, 1, 1, 1, 1, 1]
        use_temp_convs_5 = [0, 1, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)

    # C2D, ResNet101.
    if arc_type == 3:
        use_temp_convs_1 = [0]
        temp_strides_1 = [1]
        use_temp_convs_2 = [0, 0, 0]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [0, 0, 0, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [0, ] * 23
        temp_strides_4 = [1, ] * 23
        use_temp_convs_5 = [0, 0, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)

    # I3D, ResNet101.
    if arc_type == 4:
        use_temp_convs_1 = [2]
        temp_strides_1 = [1]
        use_temp_convs_2 = [1, 1, 1]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [1, 0, 1, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = []
        for i in range(23):
            if i % 2 == 0:
                use_temp_convs_4.append(1)
            else:
                use_temp_convs_4.append(0)

        temp_strides_4 = [1, ] * 23
        use_temp_convs_5 = [0, 1, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)

    use_temp_convs_set = [
        use_temp_convs_1,
        use_temp_convs_2,
        use_temp_convs_3,
        use_temp_convs_4,
        use_temp_convs_5
    ]
    temp_strides_set = [
        temp_strides_1,
        temp_strides_2,
        temp_strides_3,
        temp_strides_4,
        temp_strides_5
    ]
    return use_temp_convs_set, temp_strides_set, pool_stride


def create_resnet_video_branch(model, data, split, 
                               lfb_infer_only, prefix, suffix=''):
    """This function defines the full model."""

    cfg.DILATIONS = 1

    group = cfg.RESNETS.NUM_GROUPS
    width_per_group = cfg.RESNETS.WIDTH_PER_GROUP
    batch_size = get_batch_size(split)
        

    logger.info(
        '--------------- ResNet_branch_{}-{} {}x{}d-{}, {}, {}, infer LFB? {}, suffix: \"{}\" ---------------'.format(
            prefix,
            cfg.MODEL.DEPTH,
            group, width_per_group,
            cfg.RESNETS.TRANS_FUNC,
            cfg.DATASET,
            split,
            lfb_infer_only,
            suffix))

    assert cfg.MODEL.DEPTH in BLOCK_CONFIG.keys(), \
        'Block config is not defined for specified model depth.'
    (n1, n2, n3, n4) = BLOCK_CONFIG[cfg.MODEL.DEPTH]

    res_block = resnet_helper._generic_residual_block_3d
    dim_inner = group * width_per_group

    crop_size = (cfg.TRAIN.CROP_SIZE if (split == 'train' and not lfb_infer_only)
                 else cfg.TEST.CROP_SIZE)

    out_spatial_dim = crop_size//(8*cfg.MODEL.CONV4_STRIDE*cfg.MODEL.CONV5_STRIDE)

    use_temp_convs_set, temp_strides_set, pool_stride = \
        obtain_arc(cfg.AVABOX.GLOBAL_BRANCH_ARC)
    logger.info("use_temp_convs_set: {}".format(use_temp_convs_set))
    logger.info("temp_strides_set: {}".format(temp_strides_set))

    conv_blob = model.ConvNd(
        data,
        prefix+'_conv1',
        3,
        64,
        [1 + use_temp_convs_set[0][0] * 2, 7, 7],
        strides=[temp_strides_set[0][0], 2, 2],
        pads=[use_temp_convs_set[0][0], 3, 3] * 2,
        weight_init=('MSRAFill', {}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=1
    )

    test_mode = False if split not in ['test', 'val'] else True
    
    cfg.TEST_MODE = test_mode
    
    if cfg.MODEL.USE_AFFINE:
        bn_blob = model.AffineNd(conv_blob, prefix+'_res_conv1_bn', 64)
    else:
        bn_blob = model.SpatialBN(
            conv_blob, prefix+'_res_conv1_bn', 64, epsilon=cfg.MODEL.BN_EPSILON,
            momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
        )
    relu_blob = model.Relu(bn_blob, bn_blob)
    max_pool = model.MaxPool(
        relu_blob,
        prefix+'_pool1',
        kernels=[1, 3, 3],
        strides=[1, 2, 2],
        pads=[0, 1, 1] * 2
    )

    if cfg.MODEL.DEPTH in [50, 101]:
        blob_in, dim_in = resnet_helper.res_stage_nonlocal(
            model,
            res_block,
            max_pool,
            64,
            256,
            stride=1,
            num_blocks=n1,
            prefix=prefix+'_res2',
            dim_inner=dim_inner, group=group,
            use_temp_convs=use_temp_convs_set[1],
            temp_strides=temp_strides_set[1]
        )

        layer_mod = cfg.NONLOCAL.LAYER_MOD
        if cfg.MODEL.DEPTH == 101:
            layer_mod = 2
        if not cfg.NONLOCAL.CONV3_NONLOCAL:
            layer_mod = 1000

        blob_in = model.MaxPool(
            blob_in,
            prefix+'_pool2',
            kernels=[2, 1, 1],
            strides=[2, 1, 1],
            pads=[0, 0, 0] * 2
        )

        if not cfg.MODEL.USE_AFFINE:
            blob_in, dim_in = resnet_helper.res_stage_nonlocal(
                model,
                res_block,
                blob_in,
                dim_in,
                512,
                stride=2,
                num_blocks=n2,
                prefix=prefix+'_res3',
                dim_inner=dim_inner * 2,
                group=group,
                use_temp_convs=use_temp_convs_set[2],
                temp_strides=temp_strides_set[2],
                batch_size=batch_size,
                nonlocal_name=prefix+'_nonlocal_conv3',
                nonlocal_mod=layer_mod
            )
        else:
            blob_in, dim_in = resnet_helper.res_stage_nonlocal_group(
                model,
                res_block,
                blob_in,
                dim_in,
                512,
                stride=2,
                num_blocks=n2,
                prefix=prefix+'_res3',
                dim_inner=dim_inner * 2,
                group=group,
                use_temp_convs=use_temp_convs_set[2],
                temp_strides=temp_strides_set[2],
                batch_size=batch_size,
                pool_stride=pool_stride,
                spatial_dim=(int(crop_size / 8)),
                group_size=4,
                nonlocal_name=prefix+'_nonlocal_conv3',
                nonlocal_mod=layer_mod
            )

        layer_mod = cfg.NONLOCAL.LAYER_MOD
        if cfg.MODEL.DEPTH == 101:
            layer_mod = layer_mod * 4 - 1
        if not cfg.NONLOCAL.CONV4_NONLOCAL:
            layer_mod = 1000

        if cfg.MODEL.DILATIONS_IN_CONV4:
            cfg.DILATIONS = 2
        blob_in, dim_in = resnet_helper.res_stage_nonlocal(
            model,
            res_block,
            blob_in,
            dim_in,
            1024,
            stride=cfg.MODEL.CONV4_STRIDE,
            num_blocks=n3,
            prefix=prefix+'_res4',
            dim_inner=dim_inner * 4,
            group=group,
            use_temp_convs=use_temp_convs_set[3],
            temp_strides=temp_strides_set[3],
            batch_size=batch_size,
            nonlocal_name=prefix+'_nonlocal_conv4',
            nonlocal_mod=layer_mod
        )
        cfg.DILATIONS = 1
        
        if cfg.MODEL.DILATIONS_AFTER_CONV5:
            cfg.DILATIONS = 2

        if not cfg.ROI.AFTER_RES4:
            blob_in, dim_in = resnet_helper.res_stage_nonlocal(
                model, res_block, blob_in, dim_in, 2048,
                stride=cfg.MODEL.CONV5_STRIDE,
                num_blocks=n4,
                prefix=prefix+'_res5', dim_inner=dim_inner * 8, group=group,
                use_temp_convs=use_temp_convs_set[4],
                temp_strides=temp_strides_set[4])

    else:
        raise Exception("Unsupported network settings.")

    if cfg.MODEL.FREEZE_BACKBONE:
        model.StopGradient(blob_in, blob_in)
        
    return blob_in, dim_in
        
        

            
        
        
