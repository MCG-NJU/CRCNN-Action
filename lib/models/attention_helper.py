"""Attention blocks"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from core.config import config as cfg

logger = logging.getLogger(__name__)

if cfg.SE_BLOCK.XAVIER_INIT:
    weight_init = ('XavierFill', {})
else:
    weight_init = ('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD})


def spatial_attention_block(model, blob_in, 
                 dim_in, height, width, prefix, time_dim=1):
    """
    blob_in:(N, C, T, H, W)
    blob_out:(N, C, 1, 1, 1)
    """
    if cfg.SPATIAL_ATT_BLOCK.TYPE == 'cbam':
        spatial_att_max = model.net.ReduceMax(
            blob_in, prefix+'_channel_max', axes=[1])

        spatial_att_mean = model.net.ReduceMean(
            blob_in, prefix+'_channel_avg', axes=[1])

        # (N, 2, T, H, W)
        spatial_att_cat = model.net.Concat(
            [spatial_att_max, spatial_att_mean],
            [prefix+'_max_avg_cat', prefix+'_max_avg_cat_info'],
            axis=1)[0]

        SPATIAL_ATT_KERNEL=3
        pad_size= (SPATIAL_ATT_KERNEL-1)//2
       
        spatial_att = model.ConvNd(
            spatial_att_cat, 
            prefix+'_att',
            2, 1, 
            [1, SPATIAL_ATT_KERNEL, SPATIAL_ATT_KERNEL],
            strides=[1, 1, 1], 
            pads=[0, pad_size, pad_size]*2,
            weight_init=weight_init,
            bias_init=('ConstantFill', {'value': 0.}))
        
    elif cfg.SPATIAL_ATT_BLOCK.TYPE == 'conv':
        spatial_att = model.ConvNd(
            blob_in, 
            prefix+'_att',
            dim_in, 1, 
            [1, 1, 1],
            strides=[1, 1, 1], 
            pads=[0, 0, 0]*2,
            weight_init=weight_init,
            bias_init=('ConstantFill', {'value': 0.}))

    # (N, 1, T, H, W)
    if cfg.SPATIAL_ATT_BLOCK.ACT == 'sigmoid':
        spatial_att = model.Sigmoid(
            spatial_att, spatial_att+'_sigmoid')
    elif cfg.SPATIAL_ATT_BLOCK.ACT == 'softmax':
        spatial_att, _ = model.Reshape(
            spatial_att, [spatial_att+'_re', spatial_att+'_re_shape'],
            shape=[-1, time_dim*height*width]
        )
        spatial_att = model.Softmax(
            spatial_att, spatial_att+'_softmax', axis=1)
        spatial_att, _ = model.Reshape(
            spatial_att, [spatial_att+'_re', spatial_att+'_re_shape'],
            shape=[-1, 1, time_dim, height, width]
        )
    else:
        assert False

    # (N, C, T, H, W)
    blob_in = model.net.Mul([blob_in, spatial_att], prefix+'_mul')

    # (N, C, 1, 1, 1)
    blob_in = model.ReduceSum(
            blob_in, prefix+'_out', axes=[2,3,4])
    
    return blob_in


def OSME_block(model, blob_in, 
               dim_in, height, width, prefix, time_dim=1):
    """
    blob_in: (N, C, T, H, W)
    blob_out: (N, BN*D)
    """
    
    out_list = []
    for idx in range(cfg.OSME_BLOCK.BRANCH_NUM):
        # (N, C, T, H, W) -> (N, C, T, H, W)
        one_out = SE_block(model, blob_in, dim_in, 
                           height, width, 
                           prefix+'_SE%d'%idx, time_dim=time_dim)
        
        dim_out = dim_in
        #-> (N, C, 1, 1, 1)
        if cfg.OSME_BLOCK.POOL_TYPE == 'fc':
            one_out, _ = model.Reshape(
                one_out,
                [one_out+'_re', one_out+'_re_shape'],
                shape=(-1, dim_out*height*width*time_dim)
            )
            
            one_out = model.FC(
                    one_out, one_out+'_fc',
                    dim_out*height*width*time_dim, 
                    cfg.OSME_BLOCK.FC_OUT_DIM,
                    weight_init=weight_init,
                    bias_init=('ConstantFill', {'value': 0.}))
            dim_out = cfg.OSME_BLOCK.FC_OUT_DIM
            
        elif cfg.OSME_BLOCK.POOL_TYPE == 'spatial_attention':
            one_out = spatial_attention_block(model, one_out, 
                 dim_in, height, width, one_out+'_satt', time_dim)
            
        elif cfg.OSME_BLOCK.POOL_TYPE == 'avg':
            one_out = model.AveragePool(
                one_out, one_out+'_avgpooled', 
                kernels=[time_dim, height, width],
                strides=[1, 1, 1], pads=[0, 0, 0] * 2)
            
        elif cfg.OSME_BLOCK.POOL_TYPE == 'max':
            one_out = model.MaxPool(
                one_out, one_out+'_maxpooled', 
                kernels=[time_dim, height, width],
                strides=[1, 1, 1], pads=[0, 0, 0] * 2)
        else:
            assert False
            
        one_out, _ = model.Reshape(
                one_out,
                [one_out+'_re', one_out+'_re_shape'],
                shape=(-1, dim_out, 1)
            )
        out_list.append(one_out)
        
    # (N, C, BN)
    blob_out = model.net.Concat(out_list,
                [prefix + '_out', prefix + '_out_info'],
                axis=2)[0]
    
    # -> (N, BN, C)
    blob_out = model.Transpose(
            blob_out, blob_out+'_tr',
            axes=(0,2,1))
    
    if cfg.OSME_BLOCK.FUSION_TYPE == 'concat':
        # (N, BN*C)
        blob_out, _ = model.Reshape(
            blob_out, 
            [blob_out+'_re', blob_out+'_re_shape'],
            shape=(-1, cfg.OSME_BLOCK.BRANCH_NUM*dim_out)
        )
        dim_out = cfg.OSME_BLOCK.BRANCH_NUM*dim_out
    elif cfg.OSME_BLOCK.FUSION_TYPE == 'sum':
        blob_out = model.ReduceSum(
                blob_out, blob_out+'_sum', axes=[1], keepdims=False)
        dim_out = dim_out
    elif cfg.OSME_BLOCK.FUSION_TYPE == 'max':
        blob_out = model.ReduceMax(
                blob_out, blob_out+'_max', axes=[1], keepdims=False)
        dim_out = dim_out
    else:
        assert False
    
    return blob_out, dim_out


def SE_block(model, blob_in, dim_in, height, width, prefix, time_dim=1):
    """Squeeze-and-Excitation Block"""
    r = cfg.SE_BLOCK.R
    # (N,C,T,H,W) -> (N,C,1,1,1) -> (N,C//r) -> (N,C)
    if cfg.SE_BLOCK.SQUEEZE_TYPE == 'avg':
        channel_att_global = model.AveragePool(
            blob_in, prefix+'_avgpooled', 
            kernels=[time_dim, height, width],
            strides=[1, 1, 1], pads=[0, 0, 0] * 2)
    elif cfg.SE_BLOCK.SQUEEZE_TYPE == 'max':
        channel_att_global = model.MaxPool(
            blob_in, prefix+'_maxpooled', 
            kernels=[time_dim, height, width],
            strides=[1, 1, 1], pads=[0, 0, 0] * 2)
    else:
        assert False
        

    channel_att_scale = model.FC(
        channel_att_global, prefix+'_scale_fc1',
        dim_in, dim_in//r,
        weight_init=weight_init,
        bias_init=('ConstantFill', {'value': 0.}))

    channel_att_scale = model.Relu(
        channel_att_scale, channel_att_scale+'_relu')

    channel_att_scale = model.FC(
        channel_att_scale, channel_att_scale+'_fc2',
        dim_in//r, dim_in,
        weight_init=weight_init,
        bias_init=('ConstantFill', {'value': 0.}))

    # (N,C)
    channel_att_scale = model.Sigmoid(
        channel_att_scale, channel_att_scale+'_sigmoid')

    # (N,C,1,1,1)
    channel_att_scale = model.ExpandDims(
        channel_att_scale, dims=[2,3,4])

    # (N,C,T,H,W)
    blob_out = model.net.Mul(
        [blob_in, channel_att_scale], prefix+'_out')
    
    return blob_out


def CBAM_C_block(model, blob_in, dim_in, height, width, prefix):
    r = cfg.SE_BLOCK.R
    """CBAM Block Channel Attention Gate""" 
    # (N,C,H,W) -> (N,C,1,1) -> (N,C//r) -> (N,C)
    channel_att_maxpooled = model.MaxPool(
        blob_in, prefix+'_channel_att_maxpooled', kernels=[height, width],
        strides=[1, 1], pads=[0, 0] * 2)

    channel_att_maxpooled = model.FC(
        channel_att_maxpooled, prefix+'_channel_att_fc1',
        dim_in, dim_in//r,
        weight_init=('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD}),
        bias_init=('ConstantFill', {'value': 0.}))

    channel_att_maxpooled = model.Relu(
        channel_att_maxpooled, channel_att_maxpooled+'_relu')

    channel_att_maxpooled = model.FC(
        channel_att_maxpooled, prefix+'_channel_att_fc2',
        dim_in//r, dim_in,
        weight_init=('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD}),
        bias_init=('ConstantFill', {'value': 0.}))

    channel_att_maxpooled = model.Copy(
        channel_att_maxpooled, prefix+'_channel_att_maxpooled_mlp')

    # (N,C,H,W) -> (N,C,1,1) -> (N,C//r) -> (N,C)
    channel_att_avgpooled = model.AveragePool(
        blob_in, prefix+'_channel_att_avgpooled', kernels=[height, width],
        strides=[1, 1], pads=[0, 0] * 2)

    channel_att_avgpooled = model.net.FC(
        [channel_att_avgpooled, prefix+'_channel_att_fc1_w', prefix+'_channel_att_fc1_b'],
        prefix+'_channel_att_avgpooled_fc1')

    channel_att_avgpooled = model.Relu(
        channel_att_avgpooled, channel_att_avgpooled+'_relu')

    channel_att_avgpooled = model.net.FC(
        [channel_att_avgpooled, prefix+'_channel_att_fc2_w', prefix+'_channel_att_fc2_b'],
        prefix+'_channel_att_avgpooled_mlp')

    # (N,C)
    channel_attention_sum = model.net.Sum(
        [channel_att_maxpooled, channel_att_avgpooled],
        prefix+'_channel_attention_sum')

    channel_attention_sum = model.Sigmoid(
        channel_attention_sum, channel_attention_sum+'_sigmoid')

    # (N,C,1,1)
    channel_attention_sum = model.ExpandDims(
        channel_attention_sum, dims=[2,3])

    # (N,C,H,W)
    blob_out = model.net.Mul([blob_in, channel_attention_sum])
    
    return blob_out


def nl_pre_act(model, x):
    """Pre-activation style non-linearity."""
    if cfg.MSNL_BLOCK.PRE_ACT_LN:
        x = model.LayerNorm(
            x, [x + "_ln",
                x + "_ln_mean",
                x + "_ln_std"])[0]
    return model.Relu(x, x + "_relu")


# init_params1 is used in theta, phi, and g.
init_params1 = {
    'weight_init': ('GaussianFill', {'std': cfg.NONLOCAL.CONV_INIT_STD}),
    'bias_init': ('ConstantFill', {'value': 0.}),
    'no_bias': cfg.NONLOCAL.NO_BIAS}

# init_params2 is used in the output 1x1 conv.
init_params2 = {
    'weight_init': ('ConstantFill', {'value': 0.}),
    'bias_init': ('ConstantFill', {'value': 0.}),
    'no_bias': cfg.NONLOCAL.NO_BIAS}

def NLCore(
        model, in_blob1, in_blob2, in_dim1, in_dim2, latent_dim,
        num_feat1, num_feat2, prefix, test_mode):
    """Core logic of non-local blocks."""

    theta = model.ConvNd(
        in_blob1, prefix + '_theta',
        in_dim1,
        latent_dim,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        **init_params1)

    phi = model.ConvNd(
        in_blob2,
        prefix + '_phi',
        in_dim2,
        latent_dim,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        **init_params1)

    g = model.ConvNd(
        in_blob2,
        prefix + '_g',
        in_dim2,
        latent_dim,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        **init_params1)

    theta, theta_shape_5d = model.Reshape(
        theta,
        [theta + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else theta,
         theta + '_shape5d'],
        shape=(-1, latent_dim, num_feat1))

    phi, phi_shape_5d = model.Reshape(
        phi,
        [phi + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else phi,
         phi + '_shape5d'],
        shape=(-1, latent_dim, num_feat2))

    g, g_shape_5d = model.Reshape(
        g,
        [g + '_re',
         g + '_shape5d'],
        shape=(-1, latent_dim, num_feat2))

    # (N, C, num_feat1), (N, C, num_feat2) -> (N, num_feat1, num_feat2)
    theta_phi = model.net.BatchMatMul(
        [theta, phi], prefix + '_affinity', trans_a=1)

    if cfg.MSNL_BLOCK.SCALE:
        theta_phi = model.Scale(
            theta_phi, theta_phi, scale=latent_dim**-.5)

    p = model.Softmax(
        theta_phi, theta_phi + '_prob', engine='CUDNN', axis=2)

    # (N, C, num_feat2), (N, num_feat1, num_feat2) -> (B, C, num_feat1)
    t = model.net.BatchMatMul([g, p], prefix + '_y', trans_b=1)

    blob_out, t_shape = model.Reshape(
        [t, theta_shape_5d],
        [t + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else t,
            t + '_shape3d'])

    if cfg.MSNL_BLOCK.PRE_ACT:
        blob_out = nl_pre_act(model, blob_out)

    blob_out = model.ConvNd(
        blob_out, prefix + '_out',
        latent_dim,
        in_dim1,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        **init_params2)

    if not cfg.MSNL_BLOCK.PRE_ACT:
        blob_out = model.LayerNorm(
            blob_out,
            [prefix + "_ln", prefix + "_ln_mean", prefix + "_ln_std"])[0]

    if cfg.MSNL_BLOCK.OUTPUT_DROPOUT_ON and not test_mode:
        blob_out = model.Dropout(
            blob_out, blob_out + '_drop',
            ratio=cfg.MSNL_BLOCK.DROPOUT_RATE, is_test=False)

    return blob_out


def NLCore_T1_unshared(
        model, in_blob1, in_blob2, in_dim1, in_dim2, latent_dim,
        num_feat1, num_feat2, prefix, test_mode, branch_num):
    """Core logic of non-local blocks."""
    """Transform of theta is unshared"""

    all_theta = []
    for i in range(branch_num):
        theta = model.ConvNd(
            in_blob1, prefix + '_theta' +'_branch%d'%i,
            in_dim1,
            latent_dim,
            [1, 1, 1],
            strides=[1, 1, 1],
            pads=[0, 0, 0] * 2,
            **init_params1)
        all_theta.append(theta)
        
    theta = model.net.Concat(all_theta,
                [prefix + '_theta', prefix + '_info'],
                axis=2)[0]
    num_feat1 = num_feat1 * branch_num

    phi = model.ConvNd(
        in_blob2,
        prefix + '_phi',
        in_dim2,
        latent_dim,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        **init_params1)

    g = model.ConvNd(
        in_blob2,
        prefix + '_g',
        in_dim2,
        latent_dim,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        **init_params1)

    theta, theta_shape_5d = model.Reshape(
        theta,
        [theta + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else theta,
         theta + '_shape5d'],
        shape=(-1, latent_dim, num_feat1))

    phi, phi_shape_5d = model.Reshape(
        phi,
        [phi + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else phi,
         phi + '_shape5d'],
        shape=(-1, latent_dim, num_feat2))

    g, g_shape_5d = model.Reshape(
        g,
        [g + '_re',
         g + '_shape5d'],
        shape=(-1, latent_dim, num_feat2))

    # (N, C, num_feat1), (N, C, num_feat2) -> (N, num_feat1, num_feat2)
    theta_phi = model.net.BatchMatMul(
        [theta, phi], prefix + '_affinity', trans_a=1)

    if cfg.MSNL_BLOCK.SCALE:
        theta_phi = model.Scale(
            theta_phi, theta_phi, scale=latent_dim**-.5)

    if cfg.MSNL_BLOCK.ATT_ACT == 'softmax':
        p = model.Softmax(
            theta_phi, theta_phi + '_prob', engine='CUDNN', axis=2)
    elif cfg.MSNL_BLOCK.ATT_ACT == 'sigmoid':
        p = model.Sigmoid(
            theta_phi, theta_phi + '_prob'
        )
    else:
        assert False

    # (N, C, num_feat2), (N, num_feat1, num_feat2) -> (B, C, num_feat1)
    t = model.net.BatchMatMul([g, p], prefix + '_y', trans_b=1)

    blob_out, t_shape = model.Reshape(
        [t, theta_shape_5d],
        [t + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else t,
            t + '_shape3d'])

    if cfg.MSNL_BLOCK.PRE_ACT:
        blob_out = nl_pre_act(model, blob_out)

    blob_out = model.ConvNd(
        blob_out, prefix + '_out',
        latent_dim,
        in_dim1,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        **init_params2)

    if not cfg.MSNL_BLOCK.PRE_ACT:
        blob_out = model.LayerNorm(
            blob_out,
            [prefix + "_ln", prefix + "_ln_mean", prefix + "_ln_std"])[0]

    if cfg.MSNL_BLOCK.OUTPUT_DROPOUT_ON and not test_mode:
        blob_out = model.Dropout(
            blob_out, blob_out + '_drop',
            ratio=cfg.MSNL_BLOCK.DROPOUT_RATE, is_test=False)

    return blob_out


def MSNL_block(model, blob_in, dim_in, height, width, prefix, branch_num=1, time=None):
    """multi-scale non_local block"""
    # blob_in: (N, C, 7, 7)
    
    if cfg.MSNL_BLOCK.POOL_TYPE == 'max':
        pool_func = model.MaxPool
    elif cfg.MSNL_BLOCK.POOL_TYPE == 'avg':
        pool_func = model.AveragePool
    else:
        assert False
    
    if time is None:
        # -> (N, C, 1, 1)
        ms_feat_1c = pool_func(
            blob_in, prefix+'_ms_feat_1c', kernels=[height, width],
            strides=[1, 1], pads=[0, 0] * 2)
    else:
        # -> (N, C, 1, 1, 1)
        ms_feat_1c = pool_func(
            blob_in, prefix+'_ms_feat_1c', kernels=[time, height, width],
            strides=[1, 1, 1], pads=[0, 0, 0] * 2)
    
    if cfg.MSNL_BLOCK.WITH_GLOBAL:
        feat_num = 1
        all_feats = [ms_feat_1c]
    else:
        feat_num = 0
        all_feats = []
    
    for scale in cfg.MSNL_BLOCK.SCALE_LIST:
        if scale == '2_1c':
            # -> (N, C, 2, 1)
            one_scale_feats = model.MaxPool(
                blob_in, prefix+'_ms_feat_2_1c', kernels=[4, 7],
                strides=[3, 1], pads=[0, 0] * 2)
            feat_num += 2
            
        elif scale == '3_1c':
            # -> (N, C, 3, 1)
            one_scale_feats = model.MaxPool(
                blob_in, prefix+'_ms_feat_3_1c', kernels=[3, 7],
                strides=[2, 1], pads=[0, 0] * 2)
            feat_num += 3
            
        elif scale == '2_2c':
            # -> (N, C, 2, 2)
            one_scale_feats = model.MaxPool(
                blob_in, prefix+'_ms_feat_2_2c',
                kernels=[4,4], strides=[3,3], pads=[0,0] * 2)
            one_scale_feats, _ = model.Reshape(
                one_scale_feats,
                [one_scale_feats+'_re', one_scale_feats+'_re_shape'],
                shape=(-1, dim_in, 4, 1))
            feat_num += 4
            
        elif scale == '3_3c':
            # -> (N, C, 3, 3)
            one_scale_feats = model.MaxPool(
                blob_in, prefix+'_ms_feat_3_3c',
                kernels=[3,3], strides=[2,2], pads=[0,0] * 2)
            one_scale_feats, _ = model.Reshape(
                one_scale_feats,
                [one_scale_feats+'_re', one_scale_feats+'_re_shape'],
                shape=(-1, dim_in, 9, 1))
            feat_num += 9
            
        elif scale == '7_7c':
            # -> (N, C, 49, 1)
            one_scale_feats, _ = model.Reshape(
                blob_in, 
                [prefix+'_ms_feat_7_7c', prefix+'_ms_feat_49c_shape'],
                shape=(-1, dim_in, 49, 1))
            feat_num += 49
            
        elif scale == 'HW':
            # -> (N, C, HW, 1)
            one_scale_feats, _ = model.Reshape(
                blob_in, 
                [prefix+'_ms_feat_HWc', prefix+'_ms_feat_HWc_shape'],
                shape=(-1, dim_in, height*width, 1))
            feat_num += height*width
            
        elif scale == 'THW':
            assert time is not None
            # -> (N, C, THW, 1)
            one_scale_feats, _ = model.Reshape(
                blob_in, 
                [prefix+'_ms_feat_THWc', prefix+'_ms_feat_THWc_shape'],
                shape=(-1, dim_in, time*height*width, 1))
            feat_num += time*height*width
            
        else:
            raise NotImplementedError
            
        all_feats.append(one_scale_feats)
    
    
    # -> (N, C, feat_num, 1)
    ms_feat = model.net.Concat( all_feats,
                                [prefix+'_ms_feat', prefix+'_ms_feat_info'],
                                axis=2)[0]  
    
    # -> (N, C, 1, 1, 1)
    ms_feat_1c, _ = model.Reshape(
        ms_feat_1c,
        [ms_feat_1c+'_re', ms_feat_1c+'_re_shape'],
        shape=(-1, dim_in, 1, 1, 1)
    )
    
    # -> (N, C, feat_num, 1, 1)
    ms_feat, _ = model.Reshape(
        ms_feat,
        [ms_feat+'_re', ms_feat+'_re_shape'],
        shape=(-1, dim_in, feat_num, 1, 1)
    )
    
    if cfg.MSNL_BLOCK.T1_unshared:
        # -> (N, C, BN, 1, 1)
        nl_out = NLCore_T1_unshared(
            model,
            in_blob1=ms_feat_1c,
            in_blob2=ms_feat,
            in_dim1=dim_in,
            in_dim2=dim_in,
            latent_dim=cfg.MSNL_BLOCK.LATENT_DIM,
            num_feat1=1,
            num_feat2=feat_num,
            prefix=prefix+'_msnl',
            test_mode=cfg.TEST_MODE,
            branch_num=branch_num
        )
        
        # -> (N, C, 1, 1, 1) * BN
        nl_out_split = model.net.Split(
                        nl_out, [prefix+'_msnl_branch%d'%i for i in range(branch_num)],
                        axis=2)
        
        all_nl_out = []
        for nl_out in nl_out_split:
            if cfg.MSNL_BLOCK.PRE_SUM:
                nl_out = model.net.Sum(
                    [nl_out, ms_feat_1c], nl_out+'_sum')
            
            if not cfg.MSNL_BLOCK.PRE_ACT:
                nl_out = model.Relu(nl_out, nl_out+"_relu")   

            # -> (N, C, 1, 1)
            nl_out = model.Squeeze(nl_out, nl_out + '_4d', dims=[2])

            all_nl_out.append(nl_out)
        
        return all_nl_out
    
    
    all_nl_out = []
    for branch_idx in range(branch_num):
        
        A = ms_feat_1c
        
        for layer_idx in range(cfg.MSNL_BLOCK.NUM_LAYERS):
            prefix_in = prefix+'_msnl_branch%d_layer%d'%(branch_idx,layer_idx)
        
            # -> (N, C, 1, 1, 1)
            nl_out = NLCore(
                model,
                in_blob1=A,
                in_blob2=ms_feat,
                in_dim1=dim_in,
                in_dim2=dim_in,
                latent_dim=cfg.MSNL_BLOCK.LATENT_DIM,
                num_feat1=1,
                num_feat2=feat_num,
                prefix=prefix_in,
                test_mode=cfg.TEST_MODE,
            )

            if cfg.MSNL_BLOCK.PRE_SUM:
                nl_out = model.net.Sum(
                    [nl_out, A], nl_out+'_sum')

            if not cfg.MSNL_BLOCK.PRE_ACT:
                nl_out = model.Relu(nl_out, nl_out+"_relu") 
                
            A = nl_out
        
        # -> (N, C, 1, 1)
        nl_out = model.Squeeze(nl_out, nl_out + '_4d', dims=[2])
        
        all_nl_out.append(nl_out)
        
    return all_nl_out


def global_self_attention(model, blob_in, dim_in, feat_num, prefix):
    """self attention between global features and local features"""
    # blob_in: (N, C, feat_num, 1)
    # blob_out: (N, C, 1, 1)
    
    # -> (N, C, 1, 1)
    global_features = model.ReduceMax(
        blob_in, prefix+'_GA_global', axes=[2])
    
    # -> (N, C, 1, 1, 1)
    global_features, _ = model.Reshape(
        global_features,
        [global_features+'_re', global_features+'_re_shape'],
        shape=(-1, dim_in, 1, 1, 1))
    
    # -> (N, C, feat_num, 1, 1)
    local_features, _ = model.Reshape(
        blob_in,
        [prefix+'_GA_local', prefix+'_GA_local_shape'],
        shape=(-1, dim_in, feat_num, 1, 1))
    
    # -> (N, C, 1, 1, 1)
    nl_out = NLCore(
        model,
        in_blob1=global_features,
        in_blob2=local_features,
        in_dim1=dim_in,
        in_dim2=dim_in,
        latent_dim=cfg.MSNL_BLOCK.LATENT_DIM,
        num_feat1=1,
        num_feat2=feat_num,
        prefix=prefix+'_GA_nl',
        test_mode=cfg.TEST_MODE,
    )

    if cfg.MSNL_BLOCK.PRE_SUM:
        nl_out = model.net.Sum(
            [nl_out, global_features], nl_out+'_sum')

    if not cfg.MSNL_BLOCK.PRE_ACT:
        nl_out = model.Relu(nl_out, nl_out+"_relu") 
        
    # -> (N, C, 1, 1)
    nl_out = model.Squeeze(nl_out, nl_out + '_4d', dims=[2])
        
    return nl_out

