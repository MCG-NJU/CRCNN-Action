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
Helper functions for non-local neural networks
(https://arxiv.org/abs/1711.07971).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import config as cfg


def spacetime_nonlocal(
        model, blob_in, dim_in, dim_out, batch_size, prefix, dim_inner,
        is_test, max_pool_stride=2):
    """The core logic for non-local operator."""

    cur = blob_in

    theta = model.ConvNd(
        cur, prefix + '_theta',
        dim_in,
        dim_inner,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        weight_init=('GaussianFill', {'std': cfg.NONLOCAL.CONV_INIT_STD}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=cfg.NONLOCAL.NO_BIAS)

    # phi and g: half spatial size.
    # e.g., (8, 1024, 4, 14, 14) => (8, 1024, 4, 7, 7)
    if cfg.NONLOCAL.USE_MAXPOOL is True:
        max_pool = model.MaxPool(
            cur, prefix + '_pool',
            kernels=[1, max_pool_stride, max_pool_stride],
            strides=[1, max_pool_stride, max_pool_stride],
            pads=[0, 0, 0] * 2,
        )
    else:
        max_pool = cur

    phi = model.ConvNd(
        max_pool, prefix + '_phi',
        dim_in,
        dim_inner,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        weight_init=('GaussianFill', {'std': cfg.NONLOCAL.CONV_INIT_STD}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=cfg.NONLOCAL.NO_BIAS)

    g = model.ConvNd(
        max_pool, prefix + '_g',
        dim_in,
        dim_inner,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        weight_init=('GaussianFill', {'std': cfg.NONLOCAL.CONV_INIT_STD}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=cfg.NONLOCAL.NO_BIAS)

    # We use explicit batch size (to support arbitrary spacetime size),
    # e.g., (8, 1024, 4, 14, 14) => (8, 1024, 784).
    theta, theta_shape_5d = model.Reshape(
        theta, [theta + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else theta,
            theta + '_shape5d'],
        shape=(batch_size, dim_inner, -1))
    phi, phi_shape_5d = model.Reshape(
        phi, [phi + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else phi,
            phi + '_shape5d'],
        shape=(batch_size, dim_inner, -1))
    g, g_shape_5d = model.Reshape(
        g, [g + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else g,
            g + '_shape5d'],
        shape=(batch_size, dim_inner, -1))

    # e.g., (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784).
    theta_phi = model.net.BatchMatMul(
        [theta, phi], prefix + '_affinity', trans_a=1)
    if cfg.NONLOCAL.USE_SOFTMAX is True:
        if cfg.NONLOCAL.USE_SCALE is True:
            theta_phi_sc = model.Scale(
                theta_phi, theta_phi, scale=dim_inner**-.5)
        else:
            theta_phi_sc = theta_phi

        # sum(p[i, j, :]) == 1, for any i, j
        p = model.Softmax(
            theta_phi_sc, theta_phi + '_prob', engine='CUDNN', axis=2)
    else:
        ones = model.net.ConstantFill(
            [theta_phi], [theta_phi + '_ones'], value=1.)
        ones = model.net.ReduceBackSum([ones], [theta_phi + '_const'])

        zeros = model.net.ConstantFill(
            [theta_phi], [theta_phi + '_zeros'], value=0.)
        denom = model.net.Add(
            [zeros, ones], [theta_phi + '_denom'], broadcast=1, axis=0)

        model.StopGradient(denom, denom)
        p = model.net.Div([theta_phi, denom], [theta_phi + '_sc'])

    # Note: g's axis[2] corresponds to p's axis[2]
    # e.g., g(8, 1024, 784_2) * p(8, 784_1, 784_2) => (8, 1024, 784_1)
    t = model.net.BatchMatMul([g, p], prefix + '_y', trans_b=1)

    # Reshape back:
    # e.g., (8, 1024, 784) => (8, 1024, 4, 14, 14)
    t_re, t_shape = model.Reshape(
        [t, theta_shape_5d],
        [t + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else t,
            t + '_shape3d'])
    blob_out = t_re

    blob_out = model.ConvNd(
        blob_out, prefix + '_out',
        dim_inner,
        dim_out,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        weight_init=(
            ('ConstantFill', {'value': 0.})
            if cfg.NONLOCAL.USE_ZERO_INIT_CONV
            else ('GaussianFill', {'std': cfg.NONLOCAL.CONV_INIT_STD})
        ),
        bias_init=(
            'ConstantFill', {'value': 0.}), no_bias=cfg.NONLOCAL.NO_BIAS)

    if cfg.NONLOCAL.USE_BN:
        blob_out = model.SpatialBN(
            blob_out, prefix + "_bn", dim_out,
            epsilon=cfg.NONLOCAL.BN_EPSILON, momentum=cfg.NONLOCAL.BN_MOMENTUM,
            is_test=is_test
        )
        model.param_init_net.ConstantFill(
            [prefix + "_bn_s"],
            prefix + "_bn_s", value=cfg.NONLOCAL.BN_INIT_GAMMA
        )

    if cfg.NONLOCAL.USE_AFFINE is True:
        blob_out = model.AffineNd(blob_out, prefix + "_bn", dim_out)

    return blob_out


def add_nonlocal(
        model, blob_in, dim_in, dim_out, batch_size, prefix, dim_inner):
    """Non-local operator + residual connection."""

    is_test = model.split in ['test', 'val']
    blob_out = spacetime_nonlocal(
        model, blob_in, dim_in, dim_out, batch_size, prefix, dim_inner, is_test)
    blob_out = model.net.Sum([blob_in, blob_out], prefix + "_sum")
    return blob_out


def add_nonlocal_group(
        model, blob_in, dim_in, dim_out, batch_size, pool_stride,
        height, width, group_size, prefix, dim_inner):
    """
    Divide the feature maps into groups in the temporal dimension
    and perform Non-local operations inside each group.
    """

    is_test = model.split in ['test', 'val']

    group_num = int(pool_stride / group_size)
    assert(pool_stride % group_size == 0)

    if group_num > 1:
        blob_in = model.Transpose(
            blob_in, blob_in + '_trans', axes=(0, 2, 1, 3, 4))
        blob_in, blob_in_5d = model.Reshape(
            blob_in,
            [blob_in + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else blob_in,
            blob_in + '_shape5d'],
            shape=(batch_size * group_num, group_size, dim_in, height, width))
        blob_in = model.Transpose(
            blob_in, blob_in + '_trans', axes=(0, 2, 1, 3, 4))

    blob_out = spacetime_nonlocal(
        model, blob_in, dim_in, dim_out,
        batch_size * group_num, prefix, dim_inner, is_test)
    blob_out = model.net.Sum([blob_in, blob_out], prefix + "_sum")

    if group_num > 1:
        blob_out = model.Transpose(
            blob_out, blob_out + '_trans', axes=(0, 2, 1, 3, 4))
        blob_out, blob_out_shape = model.Reshape(
            [blob_out, blob_in_5d],
            [blob_out + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else blob_out,
            blob_out + '_shape5d'])
        blob_out = model.Transpose(
            blob_out, blob_out + '_trans', axes=(0, 2, 1, 3, 4))

    return blob_out
