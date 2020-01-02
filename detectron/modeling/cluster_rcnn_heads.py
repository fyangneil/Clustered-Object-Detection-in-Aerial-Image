# Copyright (c) 2017-present, Facebook, Inc.
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

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Fast R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# Cluster R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_cluster_proposals(model):
    model.CollectAndDistributeFpnClusterProposals()


def add_cluster_rcnn_outputs(model, blob_in, dim):
    """Add Cluster RoI classification and bounding box regression output ops."""
    # cluster Box classification layer
    model.FC(
        blob_in,
        'cluster_cls_score',
        dim,
        2,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cluster_cls_score', 'cluster_cls_prob', engine='CUDNN')
    # Box regression layer
    num_bbox_reg_classes = 2
    model.FC(
        blob_in,
        'cluster_bbox_pred',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )



def add_cluster_rcnn_losses(model):
    """Add losses for cluster RoI classification and bounding box regression."""
    loss_scalar = 1.0
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['cluster_cls_score', 'cluster_labels_int32'], ['cluster_cls_prob', 'loss_cluster_cls'],
        scale=model.GetLossScale() * loss_scalar
    )
    loss_bbox = model.net.SmoothL1Loss(
        [
            'cluster_bbox_pred', 'cluster_bbox_targets', 'cluster_bbox_inside_weights',
            'cluster_bbox_outside_weights'
        ],
        'loss_cluster_bbox',
        scale=model.GetLossScale() * loss_scalar
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
    model.Accuracy(['cluster_cls_prob', 'cluster_labels_int32'], 'accuracy_cluster_cls')
    model.AddLosses(['loss_cluster_cls', 'loss_cluster_bbox'])
    model.AddMetrics('accuracy_cluster_cls')
    bbox_reg_weights = cfg.MODEL.BBOX_REG_WEIGHTS
    model.AddBBoxAccuracy(
        ['cluster_bbox_pred', 'cluster_rois', 'cluster_labels_int32', 'mapped_gt_cluster_boxes'],
        ['cluster_bbox_iou', 'cluster_bbox_iou_pre'], bbox_reg_weights)
    model.AddMetrics(['cluster_bbox_iou', 'cluster_bbox_iou_pre'])
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

def add_cluster_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.Cluster_RCNN.MLP_HEAD_DIM
    cluster_roi_size = cfg.Cluster_RCNN.ROI_XFORM_RESOLUTION
    cluster_roi_feat = model.ClusterRoIFeatureTransform(
        blob_in,
        'cluster_roi_feat',
        blob_rois='cluster_rois',
        method=cfg.Cluster_RCNN.ROI_XFORM_METHOD,
        resolution=cluster_roi_size,
        sampling_ratio=cfg.Cluster_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )


    model.FC(cluster_roi_feat, 'cluster_fc6', dim_in * cluster_roi_size * cluster_roi_size, hidden_dim)
    model.Relu('cluster_fc6', 'cluster_fc6')
    model.FC('cluster_fc6', 'cluster_fc7', hidden_dim, hidden_dim)
    model.Relu('cluster_fc7', 'cluster_fc7')
    if cfg.MODEL.CASCADE_ON:
        # add stage parameters to list
        if '1' not in model.stage_params:
            model.stage_params['1'] = []
        for idx in range(-2, 0):
            model.stage_params['1'].append(model.weights[idx])
            model.stage_params['1'].append(model.biases[idx])
    return 'cluster_fc7', hidden_dim


