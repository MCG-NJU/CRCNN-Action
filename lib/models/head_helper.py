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

"""Output heads of a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from core.config import config as cfg
import models.lfb_helper as lfb_helper
import models.resnet_helper as resnet_helper

from models.attention_helper import *

logger = logging.getLogger(__name__)


def add_global_pooling(model, blob_in, dim_in, T, H, W, pool_type):
    dim_out = dim_in
    if pool_type == 'avg':
        # -> (B, C, 1, 1, 1)
        pooled = model.AveragePool(
            blob_in, blob_in + '_pooled',
            kernels=[T, H, W],
            strides=[1, 1, 1], pads=[0, 0, 0] * 2)
    
    elif pool_type == 't_avg_s_max':
        # -> (B, C, 1, H, W)
        pooled = model.AveragePool(
            blob_in, blob_in + '_tavg',
            kernels=[T, 1, 1],
            strides=[1, 1, 1], pads=[0, 0, 0] * 2
        )
        pooled = model.MaxPool(
            pooled, pooled + '_smax', 
            kernels=[1, H, W],
            strides=[1, 1, 1], pads=[0, 0, 0] * 2)
    else:
        assert False
        
    return pooled, dim_out
    

def add_basic_head(model, blob_in, dim_in, pool_stride, out_spatial_dim,
             suffix, lfb_infer_only, test_mode):
    """Add an output head for models predicting "clip-level outputs"."""
    
    perform_box_global_split = False
    if cfg.DATASET == 'avabox' and cfg.AVABOX.CONCAT_GLOBAL_FEAT:
        if not cfg.AVABOX.GLOBAL_BOX_DECOUPLE_NET:
            perform_box_global_split = True
        
    if perform_box_global_split:
        # (B*2, C, T, H, W) -> (B, 2, C, T, H, W) 
        blob_in, _ = model.Reshape(
            blob_in, 
            [blob_in+'_boxglobal', blob_in+'_boxglobal_shape'],
            shape=(-1, 2, dim_in, pool_stride, out_spatial_dim, out_spatial_dim))
        
        # (B, 2, C, T, H, W) ->  (B, 1, C, T, H, W)*2
        box_feat, global_feat = model.net.Split(
                                    blob_in, 
                                    ['avabox_box_feat','avabox_global_feat'], 
                                    axis=1)
        
        # (B, C, T, H, W)
        box_feat = model.net.Squeeze(box_feat, box_feat+'_sq', dims=[1])
        global_feat = model.net.Squeeze(global_feat, global_feat+'_sq', dims=[1])

        # (B, C, 1, 1, 1)
        box_feat, dim_in = add_global_pooling(model, box_feat, dim_in,
                                      pool_stride, out_spatial_dim, out_spatial_dim,
                                      cfg.MODEL.GLOBAL_POOLING_TYPE)
        
        
        if cfg.AVABOX.CONCAT_GLOBAL_TYPE == 'concat':
            # (B, C, 1, 1, 1)
            global_feat, _ = add_global_pooling(model, global_feat, dim_in,
                                      pool_stride, out_spatial_dim, out_spatial_dim,
                                      cfg.MODEL.GLOBAL_POOLING_TYPE)
            
            if not cfg.AVABOX.OTHER_LOSS_UNSHARED_DP:
                if cfg.TRAIN.DROPOUT_RATE > 0 and not test_mode:
                    box_feat = model.Dropout(
                        box_feat, box_feat + '_dropout',
                        ratio=cfg.TRAIN.DROPOUT_RATE, is_test=False)
                    global_feat = model.Dropout(
                        global_feat, global_feat + '_dropout',
                        ratio=cfg.TRAIN.DROPOUT_RATE, is_test=False)
            
            cfg.box_blob = box_feat
            cfg.global_blob = global_feat
            
            # (B, 2*C, 1, 1, 1)
            pooled, _ = model.net.Concat([box_feat, global_feat],
                          ['avabox_cat', 'avabox_cat_info'], axis=1)
            dim_in = dim_in * 2
        
        elif cfg.AVABOX.CONCAT_GLOBAL_TYPE == 'nl_concat':
            num_lfb_feat = pool_stride*out_spatial_dim*out_spatial_dim
            global_feat, _ = model.Reshape(
                global_feat, 
                [global_feat+'_re', global_feat+'_re_shape'],
                shape=(-1, dim_in, num_lfb_feat, 1, 1)
            )
            fbo_out, fbo_out_dim = lfb_helper.add_fbo_nl_head_withlfb(
                model, box_feat, global_feat, 
                dim_in, num_lfb_feat, 
                test_mode, suffix)
            
            pooled, _ = model.net.Concat([box_feat, fbo_out],
                          ['avabox_cat', 'avabox_cat_info'], axis=1)
            
            dim_in = dim_in + fbo_out_dim
        else:
            assert False
    

    else:
        pooled, dim_in = add_global_pooling(model, blob_in, dim_in,
                                    pool_stride, out_spatial_dim, out_spatial_dim,
                                    cfg.MODEL.GLOBAL_POOLING_TYPE)
            
    pooled = model.Copy(pooled, 'box_pooled')
    
    all_heads = [pooled]
    new_dim_in = [dim_in]

    if cfg.LFB.ENABLED and not lfb_infer_only:

        if cfg.DATASET == 'avabox':
            num_lfb_feat = cfg.LFB.WINDOW_SIZE * cfg.AVA.LFB_MAX_NUM_FEAT_PER_STEP
        else:
            num_lfb_feat = cfg.LFB.WINDOW_SIZE
        
        fbo_out, fbo_out_dim = lfb_helper.add_fbo_head(
            model, pooled, dim_in,
            num_lfb_feat=num_lfb_feat,
            test_mode=test_mode, suffix=suffix)

        all_heads.append(fbo_out)
        new_dim_in.append(fbo_out_dim)

    return (model.net.Concat(all_heads,
                             ['pool5', 'pool5_concat_info'],
                             axis=1)[0],
            sum(new_dim_in))


def add_roi_head(model, blob_in, dim_in, pool_stride, out_spatial_dim,
             suffix, lfb_infer_only, test_mode):
    """Add an output head for models predicting "box-level outputs"."""

    roi_pool_func_dict={
        'roi_pool': roi_pool,
        'roi_pool_avgpool': roi_pool_avgpool,
        'roi_pood_with_res5': roi_pood_with_res5,
        'full_image_pool': full_image_pool
    }
    roi_pool_func=roi_pool_func_dict[cfg.ROI.TYPE]
    
    # (B, 2048, 16, 14, 14) -> (N, 2048, 1, 1, 1)
    roi_feat = roi_pool_func(model, blob_in, dim_in, out_spatial_dim, suffix)
    
    if isinstance(roi_feat, tuple):
        dim_in = roi_feat[1]
        roi_feat = roi_feat[0]
        
    all_heads = [roi_feat]
    new_dim_in = [dim_in]

    if cfg.LFB.ENABLED and not (lfb_infer_only and not cfg.FEATURE_MAP_LOADER.ENALBE):

        fbo_out, fbo_out_dim = lfb_helper.add_fbo_head(
            model, roi_feat, dim_in,
            num_lfb_feat=(cfg.LFB.WINDOW_SIZE
                          * cfg.AVA.LFB_MAX_NUM_FEAT_PER_STEP),
            test_mode=test_mode, suffix=suffix)

        all_heads.append(fbo_out)
        new_dim_in.append(fbo_out_dim)

    return (model.net.Concat(all_heads,
                             ['pool5', 'pool5_concat_info'],
                             axis=1)[0],
            sum(new_dim_in))


def roi_pool(model, blob_in, dim_in, out_spatial_dim, suffix):
    """RoI pooling."""

    # (B, C, 16, 14, 14) -> (B, C, 1, 14, 14)
    blob_pooled = model.AveragePool(
        blob_in,
        'blob_pooled',
        kernels=[cfg.TRAIN.VIDEO_LENGTH // 2, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2
    )

    # (B, C, 1, 14, 14), (B, C, 14, 14)
    blob_pooled = model.Squeeze(blob_pooled, blob_pooled + '_4d', dims=[2])

    # (B, C, 14, 14) and (N, 5) -> (N, C, 7, 7)
    resolution = cfg.ROI.XFORM_RESOLUTION
    roi_feat = lfb_helper.RoIFeatureTransform(
        model, blob_pooled, 'roi_feat_3d',
        spatial_scale=(1.0 / cfg.ROI.SCALE_FACTOR),
        resolution=resolution,
        blob_rois='proposals{}'.format(suffix))

    if resolution > 1:
        # -> (N, C, 1, 1)
        roi_feat = model.MaxPool(
            roi_feat, 'roi_feat_1d', kernels=[resolution, resolution],
            strides=[1, 1], pads=[0, 0] * 2)

    # (N, C, 1, 1) -> (N, C, 1, 1, 1)
    roi_feat, _ = model.Reshape(
        roi_feat,
        ['box_pooled', 'roi_feat_re2_shape'],
        shape=(-1, dim_in, 1, 1, 1))

    return roi_feat


def full_image_pool(model, blob_in, dim_in, out_spatial_dim, suffix):
    roi_feat, dim_in = add_global_pooling(model, blob_in, dim_in,
                                  cfg.TRAIN.VIDEO_LENGTH // 2, 
                                  out_spatial_dim, out_spatial_dim,
                                  cfg.MODEL.GLOBAL_POOLING_TYPE)
    
    # (N, C, 1, 1) -> (N, C, 1, 1, 1)
    roi_feat, _ = model.Reshape(
        roi_feat,
        ['box_pooled', 'roi_feat_re2_shape'],
        shape=(-1, dim_in, 1, 1, 1))
    
    return roi_feat, dim_in


def roi_pood_with_res5(model, blob_in, dim_in, out_spatial_dim, suffix):
    """RoI pooling with res5."""

    # (B, C, T, 14, 14) and (N, 5) -> (N, C, T, 7, 7)
    resolution = cfg.ROI.XFORM_RESOLUTION
    time_dim = cfg.TRAIN.VIDEO_LENGTH // 2
    roi_feat = lfb_helper.Temporal_RoIFeatureTransform(
        model, blob_in, 'roi_feat_4d', 
        time_num=time_dim,
        spatial_scale=(1.0 / cfg.ROI.SCALE_FACTOR),
        resolution=resolution,
        blob_rois='proposals{}'.format(suffix))
    
    
    res_block = resnet_helper._generic_residual_block_3d
    n4 = 3
    group = cfg.RESNETS.NUM_GROUPS
    width_per_group = cfg.RESNETS.WIDTH_PER_GROUP
    dim_inner = group * width_per_group
    use_temp_convs_set = [0, 1, 0]
    temp_strides_set = [1, 1, 1]
    # add res5 -> (N, C, T, 7, 7)
    roi_feat, dim_in = resnet_helper.res_stage_nonlocal(
                model, res_block, roi_feat, dim_in, 2048, stride=1,
                num_blocks=n4,
                prefix='res5', dim_inner=dim_inner * 8, group=group,
                use_temp_convs=use_temp_convs_set,
                temp_strides=temp_strides_set)
    
    # -> (N, 2048, 1, 1, 1)
    roi_feat = model.AveragePool(
            roi_feat, roi_feat + '_pooled',
            kernels=[time_dim, resolution, resolution],
            strides=[1, 1, 1], pads=[0, 0, 0] * 2)

    # (N, C, 1, 1) -> (N, C, 1, 1, 1)
    roi_feat, _ = model.Reshape(
        roi_feat,
        ['box_pooled', 'roi_feat_re2_shape'],
        shape=(-1, dim_in, 1, 1, 1))

    return roi_feat, dim_in


def roi_pool_avgpool(model, blob_in, dim_in, out_spatial_dim, suffix):
    """RoI pooling using avg pooling."""

    # (B, C, 16, 14, 14) -> (B, C, 1, 14, 14)
    blob_pooled = model.AveragePool(
        blob_in,
        'blob_pooled',
        kernels=[cfg.TRAIN.VIDEO_LENGTH // 2, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2
    )

    # (B, C, 1, 14, 14), (B, C, 14, 14)
    blob_pooled = model.Squeeze(blob_pooled, blob_pooled + '_4d', dims=[2])

    # (B, C, 14, 14) and (N, 5) -> (N, C, 7, 7)
    resolution = cfg.ROI.XFORM_RESOLUTION
    roi_feat = lfb_helper.RoIFeatureTransform(
        model, blob_pooled, 'roi_feat_3d',
        spatial_scale=(1.0 / cfg.ROI.SCALE_FACTOR),
        resolution=resolution,
        blob_rois='proposals{}'.format(suffix))

    if resolution > 1:
        # -> (N, C, 1, 1)
        roi_feat = model.AveragePool(
            roi_feat, 'roi_feat_1d', kernels=[resolution, resolution],
            strides=[1, 1], pads=[0, 0] * 2)

    # (N, C, 1, 1) -> (N, C, 1, 1, 1)
    roi_feat, _ = model.Reshape(
        roi_feat,
        ['box_pooled', 'roi_feat_re2_shape'],
        shape=(-1, dim_in, 1, 1, 1))

    return roi_feat

