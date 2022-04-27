# ------------------------------
# final version
# ------------------------------
import numpy as np
import torch
from mmcv.cnn import Scale, bias_init_with_prob, normal_init, ConvModule
from mmcv.runner import force_fp32
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr, limit_period
from mmdet3d.core.bbox import points_cam2img, points_img2cam
from mmdet.core import distance2bbox, multi_apply
from mmdet.models.builder import HEADS, build_loss
from .fcos_mono3d_head import FCOSMono3DHead
import math

@HEADS.register_module()
class FSGMono3dHead(FCOSMono3DHead):
    def __init__(self,
                 with_ht=True,
                 with_fs_yaw=True,
                 with_size_prior=True,
                 depth_layer_ranges=None,
                 with_direct_depth=True,
                 use_depth_classifier=True,
                 use_onlyreg_proj=False,
                 weight_dim=-1,
                 weight_branch=((256, ), ),
                 depth_branch=(64, ),
                 depth_range=(0, 70),
                 depth_unit=10,
                 division='uniform',
                 depth_bins=8,
                 loss_depth=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_bbox2d=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_consistency=dict(type='GIoULoss', loss_weight=1.0),
                 pred_velo=False,
                 pred_bbox2d=True,
                 pred_keypoints=False,
                 bbox_coder=dict(
                     type='PGDBBoxCoder',
                     base_depths=((28.01, 16.32), ),
                     base_dims=((0.8, 1.73, 0.6), (1.76, 1.73, 0.6),
                                (3.9, 1.56, 1.6)),
                     code_size=7,
                     depth_range=(0, 70),
                     depth_unit=10,
                     division='uniform',
                     depth_bins=8),
                 base_depths=((28.01, 16.32), ),
                 with_pts_depth=None,
                 with_pts_uncertainty=None,
                 fsg_center_sample_radius=2.0, # 1.5 or 2.5
                 kpt_coef = 0.2,
                 with_ensemble_proj=False,
                 **kwargs):
        self.with_ht = with_ht
        self.with_fs_yaw = with_fs_yaw
        self.with_ensemble_proj = with_ensemble_proj
        self.with_size_prior = with_size_prior
        self.with_direct_depth = with_direct_depth
        self.kpt_coef = kpt_coef
        self.fsg_center_sample_radius = fsg_center_sample_radius
        self.with_pts_depth = with_pts_depth
        self.with_pts_uncertainty = with_pts_uncertainty
        self.depth_layer_ranges = depth_layer_ranges
        self.base_depths = base_depths
        self.use_depth_classifier = use_depth_classifier
        self.use_onlyreg_proj = use_onlyreg_proj
        self.depth_branch = depth_branch
        self.pred_keypoints = pred_keypoints
        self.weight_dim = weight_dim
        self.weight_branch = weight_branch
        self.weight_out_channels = []
        for weight_branch_channels in weight_branch:
            if len(weight_branch_channels) > 0:
                self.weight_out_channels.append(weight_branch_channels[-1])
            else:
                self.weight_out_channels.append(-1)
        self.depth_range = depth_range
        self.depth_unit = depth_unit
        self.division = division
        if self.division == 'uniform':
            self.num_depth_cls = int(
                (depth_range[1] - depth_range[0]) / depth_unit) + 1
            if self.num_depth_cls != depth_bins:
                print('Warning: The number of bins computed from ' +
                      'depth_unit is different from given parameter! ' +
                      'Depth_unit will be considered with priority in ' +
                      'Uniform Division.')
        else:
            self.num_depth_cls = depth_bins
        super().__init__(
            pred_bbox2d=pred_bbox2d, bbox_coder=bbox_coder, **kwargs)
        self.loss_depth = build_loss(loss_depth)
        if self.pred_bbox2d:
            self.loss_bbox2d = build_loss(loss_bbox2d)
            self.loss_consistency = build_loss(loss_consistency)
        if self.pred_keypoints:
            self.kpts_start = 9 if self.pred_velo else 7

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_sreg_convs()
        self._init_freg_convs()
        self._init_predictor()

        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1,) * len(self.centerness_branch))
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.scale_dim = 3  # only for offset, depth and size regression

        if self.pred_bbox2d:
            self.scale_dim += 1
        # if self.pred_keypoints:
        #     self.scale_dim += 1
        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)])
            for _ in self.strides
        ])
        self.sparas = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(2)])
            for _ in self.strides
        ])
        self.fparas = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(2)])
            for _ in self.strides
        ])

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_sreg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.sreg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.sreg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_freg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.freg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.freg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch,
            conv_strides=(1,) * len(self.cls_branch))
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels, 1)

        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(len(self.group_reg_dims)):
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1,) * len(reg_branch_channels)))
                self.conv_regs.append(nn.Conv2d(out_channel, reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)
                self.conv_regs.append(
                    nn.Conv2d(self.feat_channels, reg_dim, 1))

        self.conv_sreg_prevs = nn.ModuleList()
        self.conv_sregs = nn.ModuleList()
        # offset, depth, rot-direction, rot
        self.s_dims = 2
        if self.with_ht:
            self.s_dims += 1
        if self.with_fs_yaw:
            self.s_dims += 1
        for i in range(self.s_dims):
            self.conv_sreg_prevs.append(self._init_branch(conv_channels=256, conv_strides=(1,)))
        self.conv_sregs.append(nn.Conv2d(256, 6, 1)) # offset
        if self.with_pts_depth:
            self.conv_sregs.append(nn.Conv2d(256, 3, 1))  # depth
        else:
            self.conv_sregs.append(nn.Conv2d(256, 1, 1))  # depth
        if self.with_ht:
            self.conv_sregs.append(nn.Conv2d(256, 2, 1))  # rot-direction
        if self.with_fs_yaw:
            self.conv_sregs.append(nn.Conv2d(256, 3, 1))  # rot

        self.conv_freg_prevs = nn.ModuleList()
        self.conv_fregs = nn.ModuleList()
        # freg channels (256, )
        self.f_dims = 2
        if self.with_ht:
            self.f_dims += 1
        if self.with_fs_yaw:
            self.f_dims += 1
        for i in range(self.f_dims):
            self.conv_freg_prevs.append(self._init_branch(conv_channels=256, conv_strides=(1,)))
        self.conv_fregs.append(nn.Conv2d(256, 6, 1)) # offset
        if self.with_pts_depth:
            self.conv_fregs.append(nn.Conv2d(256, 3, 1)) # depth
        else:
            self.conv_fregs.append(nn.Conv2d(256, 1, 1)) # depth
        if self.with_ht:
            self.conv_fregs.append(nn.Conv2d(256, 2, 1))  # rot-direction
        if self.with_fs_yaw:
            self.conv_fregs.append(nn.Conv2d(256, 3, 1))  # rot


        if self.use_direction_classifier:
            self.conv_dir_cls_prev = self._init_branch(
                conv_channels=self.dir_branch,
                conv_strides=(1,) * len(self.dir_branch))
            self.conv_dir_cls = nn.Conv2d(self.dir_branch[-1], 2, 1)
        if self.pred_attrs:
            self.conv_attr_prev = self._init_branch(
                conv_channels=self.attr_branch,
                conv_strides=(1,) * len(self.attr_branch))
            self.conv_attr = nn.Conv2d(self.attr_branch[-1], self.num_attrs, 1)

        if self.use_depth_classifier:
            self.conv_depth_cls_prev = self._init_branch(
                conv_channels=self.depth_branch,
                conv_strides=(1, ) * len(self.depth_branch))
            self.conv_depth_cls = nn.Conv2d(self.depth_branch[-1],
                                            self.num_depth_cls, 1)
            # Data-agnostic single param lambda for local depth fusion
            self.fuse_lambda = nn.Parameter(torch.tensor(10e-5))

        if self.weight_dim != -1:
            self.conv_weight_prevs = nn.ModuleList()
            self.conv_weights = nn.ModuleList()
            for i in range(self.weight_dim):
                weight_branch_channels = self.weight_branch[i]
                weight_out_channel = self.weight_out_channels[i]
                if len(weight_branch_channels) > 0:
                    self.conv_weight_prevs.append(
                        self._init_branch(
                            conv_channels=weight_branch_channels,
                            conv_strides=(1, ) * len(weight_branch_channels)))
                    self.conv_weights.append(
                        nn.Conv2d(weight_out_channel, 1, 1))
                else:
                    self.conv_weight_prevs.append(None)
                    self.conv_weights.append(
                        nn.Conv2d(self.feat_channels, 1, 1))

        self.conv_sweight_prevs = nn.ModuleList()
        self.conv_sweights = nn.ModuleList()
        if self.with_pts_depth:
            for i in range(3):
                self.conv_sweight_prevs.append(self._init_branch(conv_channels=256, conv_strides=(1,)))
                self.conv_sweights.append(nn.Conv2d(256, 1, 1))
        else:
            for i in range(1):
                self.conv_sweight_prevs.append(self._init_branch(conv_channels=256, conv_strides=(1,)))
                self.conv_sweights.append(nn.Conv2d(256, 1, 1))


        self.conv_fweight_prevs = nn.ModuleList()
        self.conv_fweights = nn.ModuleList()
        if self.with_pts_depth:
            for i in range(3):
                self.conv_fweight_prevs.append(self._init_branch(conv_channels=256, conv_strides=(1,)))
                self.conv_fweights.append(nn.Conv2d(256, 1, 1))
        else:
            for i in range(1):
                self.conv_fweight_prevs.append(self._init_branch(conv_channels=256, conv_strides=(1,)))
                self.conv_fweights.append(nn.Conv2d(256, 1, 1))

        self.fusion_conv1 = nn.Conv2d(256 + 256, 256*4, kernel_size=3, stride=1, padding=1)
        self.fusion_conv2 = nn.Conv2d(256*4, 256, kernel_size=3, stride=1, padding=1)
        self.fusion_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fusion_conv4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.att_conv = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)
        self.att_bn = nn.BatchNorm2d(32)
        self.att_relu = nn.ReLU6(inplace=True)
        self.att_convh = nn.Conv2d(32, 256, kernel_size=1, stride=1, padding=0)
        self.att_convw = nn.Conv2d(32, 256, kernel_size=1, stride=1, padding=0)
        self.att_poolh = nn.AdaptiveAvgPool2d((None, 1))
        self.att_poolw = nn.AdaptiveAvgPool2d((1, None))

    def init_weights(self):
        """Initialize weights of the head.

        We currently still use the customized defined init_weights because the
        default init of DCN triggered by the init_cfg will init
        conv_offset.weight, which mistakenly affects the training stability.
        """
        for modules in [self.cls_convs, self.sreg_convs, self.freg_convs, self.conv_cls_prev]:
            for m in modules:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        for conv_reg_prev in self.conv_reg_prevs:
            if conv_reg_prev is None:
                continue
            for m in conv_reg_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        for conv_sreg_prev in self.conv_sreg_prevs:
            if conv_sreg_prev is None:
                continue
            for m in conv_sreg_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        for conv_freg_prev in self.conv_freg_prevs:
            if conv_freg_prev is None:
                continue
            for m in conv_freg_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        if self.use_direction_classifier:
            for m in self.conv_dir_cls_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        if self.pred_attrs:
            for m in self.conv_attr_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        for conv_reg in self.conv_regs:
            normal_init(conv_reg, std=0.01)
        for conv_sreg in self.conv_sregs:
            normal_init(conv_sreg, std=0.01)
        for conv_freg in self.conv_fregs:
            normal_init(conv_freg, std=0.01)
        if self.use_direction_classifier:
            normal_init(self.conv_dir_cls, std=0.01, bias=bias_cls)
        if self.pred_attrs:
            normal_init(self.conv_attr, std=0.01, bias=bias_cls)

        for m in self.conv_centerness_prev:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        normal_init(self.conv_centerness, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        if self.use_depth_classifier:
            for m in self.conv_depth_cls_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
            normal_init(self.conv_depth_cls, std=0.01, bias=bias_cls)

        if self.weight_dim != -1:
            for conv_weight_prev in self.conv_weight_prevs:
                if conv_weight_prev is None:
                    continue
                for m in conv_weight_prev:
                    if isinstance(m.conv, nn.Conv2d):
                        normal_init(m.conv, std=0.01)
            for conv_weight in self.conv_weights:
                normal_init(conv_weight, std=0.01)

        for conv_sweight_prev in self.conv_sweight_prevs:
            for m in conv_sweight_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        for conv_sweight in self.conv_sweights:
            normal_init(conv_sweight, std=0.01)

        for conv_fweight_prev in self.conv_fweight_prevs:
            for m in conv_fweight_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        for conv_fweight in self.conv_fweights:
            normal_init(conv_fweight, std=0.01)

        normal_init(self.fusion_conv1, std=0.01)
        normal_init(self.fusion_conv2, std=0.01)
        normal_init(self.fusion_conv3, std=0.01)
        normal_init(self.fusion_conv4, std=0.01)
        normal_init(self.att_conv, std=0.01)
        normal_init(self.att_convh, std=0.01)
        normal_init(self.att_convw, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
                dir_cls_preds (list[Tensor]): Box scores for direction class
                    predictions on each scale level, each is a 4D-tensor,
                    the channel number is num_points * 2. (bin = 2).
                weight (list[Tensor]): Location-aware weight maps on each
                    scale level, each is a 4D-tensor, the channel number is
                    num_points * 1.
                depth_cls_preds (list[Tensor]): Box scores for depth class
                    predictions on each scale level, each is a 4D-tensor,
                    the channel number is num_points * self.num_depth_cls.
                attr_preds (list[Tensor]): Attribute scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_attrs.
                centernesses (list[Tensor]): Centerness for each scale level,
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales, self.sparas, self.fparas, self.strides)

    def forward_single(self, x, scale, spara, fpara, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox and direction class
                predictions, depth class predictions, location-aware weights,
                attribute and centerness predictions of input feature maps.
        """
        cls_feat = x
        sreg_feat = x
        freg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        # extract sreg_feat freg_feat
        for sreg_layer in self.sreg_convs:
            sreg_feat = sreg_layer(sreg_feat)
        for freg_layer in self.freg_convs:
            freg_feat = freg_layer(freg_feat)

        # predict s' offset and depth
        sbbox_pred = []
        for i in range(self.s_dims):
            clone_sreg_feat = sreg_feat.clone()
            for conv_sreg_prev_layer in self.conv_sreg_prevs[i]:
                clone_sreg_feat = conv_sreg_prev_layer(clone_sreg_feat)
            sbbox_pred.append(self.conv_sregs[i](clone_sreg_feat))
        sbbox_pred = torch.cat(sbbox_pred, dim=1)
        # Just for 1-cls prior
        if self.depth_layer_ranges is not None:
            depth_layer_base = self.depth_layer_ranges[0][0] / (stride / self.strides[0])
        spara_offset, spara_depth = spara[0:2]
        clone_sbbox_pred = sbbox_pred.clone()
        # predict three corner-points and depth
        sbbox_pred[:, :6] = spara_offset(clone_sbbox_pred[:, :6]).float()
        if self.with_pts_depth:
            sbbox_pred[:, 6:9] = spara_depth(clone_sbbox_pred[:, 6:9]).float()
        else:
            sbbox_pred[:, 6:7] = spara_depth(clone_sbbox_pred[:, 6:7]).float()
        if self.depth_layer_ranges is not None:
            if self.with_pts_depth:
                sbbox_pred[:, 6:9] = (sbbox_pred.clone()[:, 6:9] * depth_layer_base).exp() # .log2()
            else:
                sbbox_pred[:, 6:7] = (sbbox_pred.clone()[:, 6:7] * depth_layer_base).exp() # .log2()
        else:
            if self.base_depths is None:
                if self.with_pts_depth:
                    sbbox_pred[:, 6:9] = sbbox_pred[:, 6:9].exp()
                else:
                    sbbox_pred[:, 6:7] = sbbox_pred[:, 6:7].exp()
            else:
                if self.with_pts_depth:
                    sbbox_pred[:, 6:9] = self.base_depths[0][0] + sbbox_pred.clone()[:, 6:9] * self.base_depths[0][1]
                else:
                    sbbox_pred[:, 6:7] = self.base_depths[0][0] + sbbox_pred.clone()[:, 6:7] * self.base_depths[0][1]

        if self.norm_on_bbox:
            if not self.training:
                # Note that this line is conducted only when testing
                sbbox_pred[:, :6] *= stride

        sweight = []
        if self.with_pts_depth:
            # sweight dim is 3
            for i in range(3):
                clone_sreg_feat = sreg_feat.clone()
                for conv_sweight_prev_layer in self.conv_sweight_prevs[i]:
                    clone_sreg_feat = conv_sweight_prev_layer(clone_sreg_feat)
                sweight.append(self.conv_sweights[i](clone_sreg_feat))
        else:
            for i in range(1):
                clone_sreg_feat = sreg_feat.clone()
                for conv_sweight_prev_layer in self.conv_sweight_prevs[i]:
                    clone_sreg_feat = conv_sweight_prev_layer(clone_sreg_feat)
                sweight.append(self.conv_sweights[i](clone_sreg_feat))
        sweight = torch.cat(sweight, dim=1)

        # predict f' offset and depth
        fbbox_pred = []
        for i in range(self.f_dims):
            clone_freg_feat = freg_feat.clone()
            for conv_freg_prev_layer in self.conv_freg_prevs[i]:
                clone_freg_feat = conv_freg_prev_layer(clone_freg_feat)
            fbbox_pred.append(self.conv_fregs[i](clone_freg_feat))
        fbbox_pred = torch.cat(fbbox_pred, dim=1)
        fpara_offset, fpara_depth = fpara[0:2]
        clone_fbbox_pred = fbbox_pred.clone()
        # predict three corner-points and depth
        fbbox_pred[:, :6] = fpara_offset(clone_fbbox_pred[:, :6]).float()
        if self.with_pts_depth:
            fbbox_pred[:, 6:9] = fpara_depth(clone_fbbox_pred[:, 6:9]).float()
        else:
            fbbox_pred[:, 6:7] = fpara_depth(clone_fbbox_pred[:, 6:7]).float()
        if self.depth_layer_ranges is not None:
            if self.with_pts_depth:
                fbbox_pred[:, 6:9] = (fbbox_pred.clone()[:, 6:9] * depth_layer_base).exp() # .log2()
            else:
                fbbox_pred[:, 6:7] = (fbbox_pred.clone()[:, 6:7] * depth_layer_base).exp() # .log2()
        else:
            if self.base_depths is None:
                if self.with_pts_depth:
                    fbbox_pred[:, 6:9] = fbbox_pred[:, 6:9].exp()
                else:
                    fbbox_pred[:, 6:7] = fbbox_pred[:, 6:7].exp()
            else:
                if self.with_pts_depth:
                    fbbox_pred[:, 6:9] = self.base_depths[0][0] + fbbox_pred.clone()[:, 6:9] * self.base_depths[0][1]
                else:
                    fbbox_pred[:, 6:7] = self.base_depths[0][0] + fbbox_pred.clone()[:, 6:7] * self.base_depths[0][1]

        if self.norm_on_bbox:
            if not self.training:
                fbbox_pred[:, :6] *= stride


        fweight = []
        if self.with_pts_depth:
            for i in range(3):
                clone_freg_feat = freg_feat.clone()
                for conv_fweight_prev_layer in self.conv_fweight_prevs[i]:
                    clone_freg_feat = conv_fweight_prev_layer(clone_freg_feat)
                fweight.append(self.conv_fweights[i](clone_freg_feat))
        else:
            for i in range(1):
                clone_freg_feat = freg_feat.clone()
                for conv_fweight_prev_layer in self.conv_fweight_prevs[i]:
                    clone_freg_feat = conv_fweight_prev_layer(clone_freg_feat)
                fweight.append(self.conv_fweights[i](clone_freg_feat))
        fweight = torch.cat(fweight, dim=1)
        # feature fusion
        reg_feat = torch.cat([sreg_feat, freg_feat], dim=1)
        reg_feat = self.fusion_conv1(reg_feat)
        reg_feat = self.fusion_conv2(reg_feat)
        reg_feat = self.fusion_conv3(reg_feat)
        reg_feat = self.fusion_conv4(reg_feat)
        rg = reg_feat.clone()
        n, c, h, w = reg_feat.size()
        reg_h = self.att_poolh(reg_feat)
        reg_w = self.att_poolw(reg_feat).permute(0, 1, 3, 2)
        reg = torch.cat([reg_h, reg_w], dim=2)
        reg = self.att_conv(reg)
        reg = self.att_bn(reg)
        reg = reg * self.att_relu(reg + 3) / 6
        reg_h, reg_w = torch.split(reg, [h, w], dim=2)
        reg_w = reg_w.permute(0, 1, 3, 2)
        weight_h = self.att_convh(reg_h).sigmoid()
        weight_w = self.att_convw(reg_w).sigmoid()
        reg_feat = rg * weight_w * weight_h
        # clone the cls_feat for reusing the feature map afterwards
        clone_cls_feat = cls_feat.clone()
        for conv_cls_prev_layer in self.conv_cls_prev:
            clone_cls_feat = conv_cls_prev_layer(clone_cls_feat)
        cls_score = self.conv_cls(clone_cls_feat)

        bbox_pred = []
        for i in range(len(self.group_reg_dims)):
            # clone the reg_feat for reusing the feature map afterwards
            clone_reg_feat = reg_feat.clone()
            if len(self.reg_branch[i]) > 0:
                for conv_reg_prev_layer in self.conv_reg_prevs[i]:
                    clone_reg_feat = conv_reg_prev_layer(clone_reg_feat)
            bbox_pred.append(self.conv_regs[i](clone_reg_feat))
        bbox_pred = torch.cat(bbox_pred, dim=1)

        dir_cls_pred = None
        if self.use_direction_classifier:
            clone_reg_feat = reg_feat.clone()
            for conv_dir_cls_prev_layer in self.conv_dir_cls_prev:
                clone_reg_feat = conv_dir_cls_prev_layer(clone_reg_feat)
            dir_cls_pred = self.conv_dir_cls(clone_reg_feat)

        attr_pred = None
        if self.pred_attrs:
            # clone the cls_feat for reusing the feature map afterwards
            clone_cls_feat = cls_feat.clone()
            for conv_attr_prev_layer in self.conv_attr_prev:
                clone_cls_feat = conv_attr_prev_layer(clone_cls_feat)
            attr_pred = self.conv_attr(clone_cls_feat)

        if self.centerness_on_reg:
            clone_reg_feat = reg_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_reg_feat = conv_centerness_prev_layer(clone_reg_feat)
            centerness = self.conv_centerness(clone_reg_feat)
        else:
            clone_cls_feat = cls_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_cls_feat = conv_centerness_prev_layer(clone_cls_feat)
            centerness = self.conv_centerness(clone_cls_feat)

        bbox_pred = self.bbox_coder.decode(bbox_pred, scale, stride,
                                           self.training, depth_layer_base, cls_score, self.with_size_prior)

        # max_regress_range = stride * self.regress_ranges[0][1] / self.strides[0]
        bbox_pred = self.bbox_coder.decode_2d(bbox_pred, scale, stride,
                                              self.training,
                                              False,
                                              self.pred_bbox2d)

        depth_cls_pred = None
        if self.use_depth_classifier:
            clone_reg_feat = reg_feat.clone()
            for conv_depth_cls_prev_layer in self.conv_depth_cls_prev:
                clone_reg_feat = conv_depth_cls_prev_layer(clone_reg_feat)
            depth_cls_pred = self.conv_depth_cls(clone_reg_feat)

        weight = None
        if self.weight_dim != -1:
            weight = []
            for i in range(self.weight_dim):
                clone_reg_feat = reg_feat.clone()
                if len(self.weight_branch[i]) > 0:
                    for conv_weight_prev_layer in self.conv_weight_prevs[i]:
                        clone_reg_feat = conv_weight_prev_layer(clone_reg_feat)
                weight.append(self.conv_weights[i](clone_reg_feat))
            weight = torch.cat(weight, dim=1)

        return cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, \
            attr_pred, centerness, sbbox_pred, sweight, fbbox_pred, fweight

    def get_proj_bbox2d(self,
                        bbox_preds,
                        pos_dir_cls_preds,
                        labels_3d,
                        bbox_targets_3d,
                        pos_points,
                        pos_inds,
                        img_metas,
                        pos_depth_cls_preds=None,
                        ensemble_depth=None,
                        pos_weights=None,
                        pos_cls_scores=None,
                        with_kpts=False,
                        with_fs_kpts=False):
        """Decode box predictions and get projected 2D attributes.

        Args:
            bbox_preds (list[Tensor]): Box predictions for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            pos_dir_cls_preds (Tensor): Box scores for direction class
                predictions of positive boxes on all the scale levels in shape
                (num_pos_points, 2).
            labels_3d (list[Tensor]): 3D box category labels for each scale
                level, each is a 4D-tensor.
            bbox_targets_3d (list[Tensor]): 3D box targets for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            pos_points (Tensor): Foreground points.
            pos_inds (Tensor): Index of foreground points from flattened
                tensors.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            pos_depth_cls_preds (Tensor, optional): Probabilistic depth map of
                positive boxes on all the scale levels in shape
                (num_pos_points, self.num_depth_cls). Defaults to None.
            pos_weights (Tensor, optional): Location-aware weights of positive
                boxes in shape (num_pos_points, self.weight_dim). Defaults to
                None.
            pos_cls_scores (Tensor, optional): Classification scores of
                positive boxes in shape (num_pos_points, self.num_classes).
                Defaults to None.
            with_kpts (bool, optional): Whether to output keypoints targets.
                Defaults to False.

        Returns:
            tuple[Tensor]: Exterior 2D boxes from projected 3D boxes,
                predicted 2D boxes and keypoint targets (if necessary).
        """
        views = [np.array(img_meta['cam2img']) for img_meta in img_metas]
        num_imgs = len(img_metas)
        img_idx = []
        for label in labels_3d:
            for idx in range(num_imgs):
                img_idx.append(
                    labels_3d[0].new_ones(int(len(label) / num_imgs)) * idx)
        img_idx = torch.cat(img_idx)
        pos_img_idx = img_idx[pos_inds]

        flatten_strided_bbox_preds = []
        flatten_strided_bbox2d_preds = []
        flatten_bbox_targets_3d = []
        flatten_strides = []

        for stride_idx, bbox_pred in enumerate(bbox_preds):
            flatten_bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
                -1, sum(self.group_reg_dims))
            flatten_bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_bbox_pred[:, -4:] *= self.strides[stride_idx]
            flatten_strided_bbox_preds.append(
                flatten_bbox_pred[:, :self.bbox_coder.bbox_code_size])
            flatten_strided_bbox2d_preds.append(flatten_bbox_pred[:, -4:])

            bbox_target_3d = bbox_targets_3d[stride_idx].clone()
            bbox_target_3d[:, :2] *= self.strides[stride_idx]
            bbox_target_3d[:, -4:] *= self.strides[stride_idx]
            flatten_bbox_targets_3d.append(bbox_target_3d)

            flatten_stride = flatten_bbox_pred.new_ones(
                *flatten_bbox_pred.shape[:-1], 1) * self.strides[stride_idx]
            flatten_strides.append(flatten_stride)

        flatten_strided_bbox_preds = torch.cat(flatten_strided_bbox_preds)
        flatten_strided_bbox2d_preds = torch.cat(flatten_strided_bbox2d_preds)
        flatten_bbox_targets_3d = torch.cat(flatten_bbox_targets_3d)
        flatten_strides = torch.cat(flatten_strides)
        pos_strided_bbox_preds = flatten_strided_bbox_preds[pos_inds]
        pos_strided_bbox2d_preds = flatten_strided_bbox2d_preds[pos_inds]
        pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
        pos_strides = flatten_strides[pos_inds]

        pos_decoded_bbox2d_preds = distance2bbox(pos_points,
                                                 pos_strided_bbox2d_preds)

        pos_strided_bbox_preds[:, :2] = \
            pos_points - pos_strided_bbox_preds[:, :2]
        pos_bbox_targets_3d[:, :2] = \
            pos_points - pos_bbox_targets_3d[:, :2]

        # what's the significance for calculating depth
        # depth will be fixed when computing re-project 3D bboxes
        if self.use_depth_classifier and (not self.use_onlyreg_proj):
            pos_strided_bbox_preds[:, 2] = ensemble_depth[:, 0]

        box_corners_in_image = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))
        box_corners_in_image_gt = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))
        cam_paras = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 2))

        for idx in range(num_imgs):
            mask = (pos_img_idx == idx)
            if pos_strided_bbox_preds[mask].shape[0] == 0:
                continue
            cam2img = torch.eye(
                4,
                dtype=pos_strided_bbox_preds.dtype,
                device=pos_strided_bbox_preds.device)
            view_shape = views[idx].shape
            cam2img[:view_shape[0], :view_shape[1]] = \
                pos_strided_bbox_preds.new_tensor(views[idx])

            centers2d_preds = pos_strided_bbox_preds.clone()[mask, :2]
            centers2d_targets = pos_bbox_targets_3d.clone()[mask, :2]
            # reprojection from img to cam
            centers3d_targets = points_img2cam(pos_bbox_targets_3d[mask, :3],
                                               views[idx])

            # use predicted depth to re-project the 2.5D centers
            pos_strided_bbox_preds[mask, :3] = points_img2cam(
                pos_strided_bbox_preds[mask, :3], views[idx])
            pos_bbox_targets_3d[mask, :3] = centers3d_targets

            # depth fixed when computing re-project 3D bboxes
            if not self.with_ensemble_proj:
                pos_strided_bbox_preds[mask, 2] = pos_bbox_targets_3d.clone()[mask, 2]
            # depth fixed when computing re-project 3D bboxes
            if self.with_ensemble_proj:
                pos_strided_bbox_preds[mask, 2] = ensemble_depth[mask, 0]

            # decode yaws
            if self.use_direction_classifier:
                pos_dir_cls_scores = torch.max(
                    pos_dir_cls_preds[mask], dim=-1)[1]
                pos_strided_bbox_preds[mask] = self.bbox_coder.decode_yaw(
                    pos_strided_bbox_preds[mask], centers2d_preds,
                    pos_dir_cls_scores, self.dir_offset, cam2img)
            pos_bbox_targets_3d[mask, 6] = torch.atan2(
                centers2d_targets[:, 0] - cam2img[0, 2],
                cam2img[0, 0]) + pos_bbox_targets_3d[mask, 6]

            corners = img_metas[0]['box_type_3d'](
                pos_strided_bbox_preds[mask],
                box_dim=self.bbox_coder.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image[mask] = points_cam2img(corners, cam2img)

            corners_gt = img_metas[0]['box_type_3d'](
                pos_bbox_targets_3d[mask, :self.bbox_code_size],
                box_dim=self.bbox_coder.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image_gt[mask] = points_cam2img(corners_gt, cam2img)
            cam_paras[mask, 0] = cam2img[0, 0]
            cam_paras[mask, 1] = cam2img[0, 2]
        pos_ht_cls_targets = self.get_ht_target(pos_bbox_targets_3d, one_hot=False)

        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        proj_bbox2d_preds = torch.cat([minxy, maxxy], dim=1)

        outputs = (proj_bbox2d_preds, pos_decoded_bbox2d_preds)

        if with_kpts:
            # norm_strides = pos_strides * self.regress_ranges[0][1] / self.strides[0]
            # encode norm_strides without regress_ranges
            norm_strides = pos_strides
            kpts_targets = box_corners_in_image_gt - pos_points[..., None, :]
            kpts_targets = kpts_targets.view(
                (*pos_strided_bbox_preds.shape[:-1], 16))
            kpts_targets /= norm_strides

            kpts_preds = box_corners_in_image - pos_points[..., None, :]
            kpts_preds = kpts_preds.view(
                (*pos_strided_bbox_preds.shape[:-1], 16))
            kpts_preds /= norm_strides

            outputs += (kpts_preds, kpts_targets, )

        # with_fs_kpts = True
        if with_fs_kpts:
            spts_targets, fpts_targets = [], []
            fspts_targets = box_corners_in_image_gt - pos_points[..., None, :]
            for i in range(pos_bbox_targets_3d.shape[0]):
                alpha = pos_bbox_targets_3d[i, 6] - math.atan(pos_bbox_targets_3d[i, 0] / pos_bbox_targets_3d[i, 2])
                alpha = alpha - torch.floor(alpha / (2*np.pi)) * (2*np.pi)
                if alpha > 0 and alpha <= np.pi/2:
                    spts, fpts = [4, 7, 3], [4, 7, 6] # [5, 1, 2], [5, 1, 0]
                elif alpha > np.pi/2 and alpha <= np.pi:
                    spts, fpts = [5, 6, 2], [5, 6, 7] # [4, 0, 3], [4, 0, 1]
                elif alpha > 3*np.pi/2 and alpha <= 2*np.pi:
                    spts, fpts = [0, 3, 7], [0, 3, 2] # [6, 2, 1], [6, 2, 3]
                elif alpha > np.pi and alpha <= 3*np.pi/2:
                    spts, fpts = [1, 2, 6], [1, 2, 3] # [7, 3, 0], [7, 3, 2]
                spts_targets.append(fspts_targets[i:i+1, spts, :])
                fpts_targets.append(fspts_targets[i:i+1, fpts, :])
            spts_targets = torch.cat(spts_targets, dim=0)
            fpts_targets = torch.cat(fpts_targets, dim=0)

            spts_yaw_targets = -torch.atan2(spts_targets[..., 0] - cam_paras[..., None, 1], cam_paras[..., None, 0]) + pos_bbox_targets_3d[..., None, 6]
            fpts_yaw_targets = -torch.atan2(fpts_targets[..., 0] - cam_paras[..., None, 1], cam_paras[..., None, 0]) + pos_bbox_targets_3d[..., None, 6]

            spts_targets = spts_targets.view((*pos_strided_bbox_preds.shape[:-1], 6))
            fpts_targets = fpts_targets.view((*pos_strided_bbox_preds.shape[:-1], 6))

            spts_targets /= norm_strides
            fpts_targets /= norm_strides

            outputs += (spts_targets, fpts_targets)

        outputs += (pos_ht_cls_targets, spts_yaw_targets, fpts_yaw_targets, cam_paras)

        return outputs

    @staticmethod
    def get_ht_target(reg_targets, num_bins=2, one_hot=True):
        rot_gt = reg_targets[..., 6]
        offset_rot = limit_period(rot_gt, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(
                *list(dir_cls_targets.shape),
                num_bins,
                dtype=reg_targets.dtype,
                device=dir_cls_targets.device)
            dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_pos_predictions(self, bbox_preds, dir_cls_preds, depth_cls_preds,
                            weights, attr_preds, centernesses, pos_inds,
                            img_metas):
        """Flatten predictions and get positive ones.

        Args:
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            depth_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * self.num_depth_cls.
            attr_preds (list[Tensor]): Attribute scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_attrs.
            centernesses (list[Tensor]): Centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            pos_inds (Tensor): Index of foreground points from flattened
                tensors.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple[Tensor]: Box predictions, direction classes, probabilistic
                depth maps, location-aware weight maps, attributes and
                centerness predictions.
        """
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds
        ]
        flatten_dir_cls_preds = [
            dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            for dir_cls_pred in dir_cls_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        pos_depth_cls_preds = None
        if self.use_depth_classifier:
            flatten_depth_cls_preds = [
                depth_cls_pred.permute(0, 2, 3,
                                       1).reshape(-1, self.num_depth_cls)
                for depth_cls_pred in depth_cls_preds
            ]
            flatten_depth_cls_preds = torch.cat(flatten_depth_cls_preds)
            pos_depth_cls_preds = flatten_depth_cls_preds[pos_inds]

        pos_weights = None
        if self.weight_dim != -1:
            flatten_weights = [
                weight.permute(0, 2, 3, 1).reshape(-1, self.weight_dim)
                for weight in weights
            ]
            flatten_weights = torch.cat(flatten_weights)
            pos_weights = flatten_weights[pos_inds]

        pos_attr_preds = None
        if self.pred_attrs:
            flatten_attr_preds = [
                attr_pred.permute(0, 2, 3, 1).reshape(-1, self.num_attrs)
                for attr_pred in attr_preds
            ]
            flatten_attr_preds = torch.cat(flatten_attr_preds)
            pos_attr_preds = flatten_attr_preds[pos_inds]

        return pos_bbox_preds, pos_dir_cls_preds, pos_depth_cls_preds, \
            pos_weights, pos_attr_preds, pos_centerness

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds',
                  'depth_cls_preds', 'weights', 'attr_preds', 'centernesses',
                  'sbbox_preds', 'sweights', 'fbbox_preds', 'fweights'))
    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             depth_cls_preds,
             weights,
             attr_preds,
             centernesses,
             sbbox_preds,
             sweights,
             fbbox_preds,
             fweights,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             attr_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            depth_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * self.num_depth_cls.
            weights (list[Tensor]): Location-aware weights for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * self.weight_dim.
            attr_preds (list[Tensor]): Attribute scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_attrs.
            centernesses (list[Tensor]): Centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): 3D boxes ground truth with shape of
                (num_gts, code_size).
            gt_labels_3d (list[Tensor]): same as gt_labels
            centers2d (list[Tensor]): 2D centers on the image with shape of
                (num_gts, 2).
            depths (list[Tensor]): Depth ground truth with shape of
                (num_gts, ).
            attr_labels (list[Tensor]): Attributes indices of each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor]): specify which bounding boxes can
                be ignored when computing the loss. Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(depth_cls_preds) == len(weights) == len(centernesses) == \
            len(attr_preds), 'The length of cls_scores, bbox_preds, ' \
            'dir_cls_preds, depth_cls_preds, weights, centernesses, and' \
            f'attr_preds: {len(cls_scores)}, {len(bbox_preds)}, ' \
            f'{len(dir_cls_preds)}, {len(depth_cls_preds)}, {len(weights)}' \
            f'{len(centernesses)}, {len(attr_preds)} are inconsistent.'
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels_3d, bbox_targets_3d, centerness_targets, attr_targets = \
            self.get_targets(
                all_level_points, gt_bboxes, gt_labels, gt_bboxes_3d,
                gt_labels_3d, centers2d, depths, attr_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores and targets
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_labels_3d = torch.cat(labels_3d)
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d)
        flatten_centerness_targets = torch.cat(centerness_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        if self.pred_attrs:
            flatten_attr_targets = torch.cat(attr_targets)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels_3d >= 0)
                    & (flatten_labels_3d < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_labels_3d,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds, pos_dir_cls_preds, pos_depth_cls_preds, pos_weights, \
            pos_attr_preds, pos_centerness = self.get_pos_predictions(
                bbox_preds, dir_cls_preds, depth_cls_preds, weights,
                attr_preds, centernesses, pos_inds, img_metas)

        # pos sbbox_preds/fbbox_preds/sweights/fweights
        if self.with_pts_depth:
            self.pts_dims = 9
            if self.with_ht:
                self.pts_dims += 2
            if self.with_fs_yaw:
                self.pts_dims += 3
            flatten_bbox_propagation_preds = [
                bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
                for bbox_pred in bbox_preds
            ]
            for stride_idx, bbox_pred in enumerate(flatten_bbox_propagation_preds):
                bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_bbox_propagation_preds = torch.cat(flatten_bbox_propagation_preds)
            pos_bbox_propagation_preds = flatten_bbox_propagation_preds[pos_inds]

            flatten_sbbox_propagation_preds = [
                sbbox_pred.permute(0, 2, 3, 1).reshape(-1, self.pts_dims)
                for sbbox_pred in sbbox_preds
            ]
            for stride_idx, bbox_pred in enumerate(flatten_sbbox_propagation_preds):
                bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_sbbox_propagation_preds = torch.cat(flatten_sbbox_propagation_preds)
            pos_sbbox_propagation_preds = flatten_sbbox_propagation_preds[pos_inds]

            flatten_fbbox_propagation_preds = [
                fbbox_pred.permute(0, 2, 3, 1).reshape(-1, self.pts_dims)
                for fbbox_pred in fbbox_preds
            ]
            for stride_idx, bbox_pred in enumerate(flatten_fbbox_propagation_preds):
                bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_fbbox_propagation_preds = torch.cat(flatten_fbbox_propagation_preds)
            pos_fbbox_propagation_preds = flatten_fbbox_propagation_preds[pos_inds]

            flatten_sbbox_preds = [
                sbbox_pred.permute(0, 2, 3, 1).reshape(-1, self.pts_dims)
                for sbbox_pred in sbbox_preds
            ]
            flatten_sbbox_preds = torch.cat(flatten_sbbox_preds)
            pos_sbbox_preds = flatten_sbbox_preds[pos_inds]
            flatten_sweights = [
                sweight.permute(0, 2, 3, 1).reshape(-1, 3)
                for sweight in sweights
            ]
            flatten_sweights = torch.cat(flatten_sweights)
            pos_sweights = flatten_sweights[pos_inds]

            flatten_fbbox_preds = [
                fbbox_pred.permute(0, 2, 3, 1).reshape(-1, self.pts_dims)
                for fbbox_pred in fbbox_preds
            ]
            flatten_fbbox_preds = torch.cat(flatten_fbbox_preds)
            pos_fbbox_preds = flatten_fbbox_preds[pos_inds]
            # focal length
            flatten_focals = flatten_fbbox_propagation_preds.new_ones(flatten_fbbox_preds.shape[0], 2)
            for i in range(num_imgs):
                flatten_focals[flatten_sbbox_preds.shape[0]*i//num_imgs: flatten_sbbox_preds.shape[0]*(i+1)//num_imgs, 0] = img_metas[i]['cam2img'][0][0]
                flatten_focals[flatten_sbbox_preds.shape[0]*i//num_imgs: flatten_sbbox_preds.shape[0]*(i+1)//num_imgs, 1] = img_metas[i]['cam2img'][1][2]
            pos_focals = flatten_focals[pos_inds]
            flatten_fweights = [
                fweight.permute(0, 2, 3, 1).reshape(-1, 3)
                for fweight in fweights
            ]
            flatten_fweights = torch.cat(flatten_fweights)
            pos_fweights = flatten_fweights[pos_inds]
        else:
            self.fs_dims = 7
            if self.with_ht:
                self.fs_dims += 2
            if self.with_fs_yaw:
                self.fs_dims += 3
            flatten_bbox_propagation_preds = [
                bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
                for bbox_pred in bbox_preds
            ]
            for stride_idx, bbox_pred in enumerate(flatten_bbox_propagation_preds):
                bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_bbox_propagation_preds = torch.cat(flatten_bbox_propagation_preds)
            pos_bbox_propagation_preds = flatten_bbox_propagation_preds[pos_inds]

            flatten_sbbox_propagation_preds = [
                sbbox_pred.permute(0, 2, 3, 1).reshape(-1, self.fs_dims)
                for sbbox_pred in sbbox_preds
            ]
            for stride_idx, bbox_pred in enumerate(flatten_sbbox_propagation_preds):
                bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_sbbox_propagation_preds = torch.cat(flatten_sbbox_propagation_preds)
            pos_sbbox_propagation_preds = flatten_sbbox_propagation_preds[pos_inds]

            flatten_fbbox_propagation_preds = [
                fbbox_pred.permute(0, 2, 3, 1).reshape(-1, self.fs_dims)
                for fbbox_pred in fbbox_preds
            ]
            for stride_idx, bbox_pred in enumerate(flatten_fbbox_propagation_preds):
                bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_fbbox_propagation_preds = torch.cat(flatten_fbbox_propagation_preds)
            pos_fbbox_propagation_preds = flatten_fbbox_propagation_preds[pos_inds]

            flatten_sbbox_preds = [
                sbbox_pred.permute(0, 2, 3, 1).reshape(-1, self.fs_dims)
                for sbbox_pred in sbbox_preds
            ]
            flatten_sbbox_preds = torch.cat(flatten_sbbox_preds)
            pos_sbbox_preds = flatten_sbbox_preds[pos_inds]
            flatten_sweights = [
                sweight.permute(0, 2, 3, 1).reshape(-1, 1)
                for sweight in sweights
            ]
            flatten_sweights = torch.cat(flatten_sweights)
            pos_sweights = flatten_sweights[pos_inds]

            flatten_fbbox_preds = [
                fbbox_pred.permute(0, 2, 3, 1).reshape(-1, self.fs_dims)
                for fbbox_pred in fbbox_preds
            ]
            flatten_fbbox_preds = torch.cat(flatten_fbbox_preds)
            pos_fbbox_preds = flatten_fbbox_preds[pos_inds]
            flatten_fweights = [
                fweight.permute(0, 2, 3, 1).reshape(-1, 1)
                for fweight in fweights
            ]
            flatten_fweights = torch.cat(flatten_fweights)
            pos_fweights = flatten_fweights[pos_inds]

        loss_dict = dict()

        if num_pos > 0:
            pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            pos_points = flatten_points[pos_inds]
            if self.pred_attrs:
                pos_attr_targets = flatten_attr_targets[pos_inds]
            if self.use_direction_classifier:
                pos_dir_cls_targets = self.get_direction_target(pos_bbox_targets_3d, self.dir_offset, one_hot=False)

            bbox_weights = pos_centerness_targets.new_ones(
                len(pos_centerness_targets), sum(self.group_reg_dims))
            equal_weights = pos_centerness_targets.new_ones(
                pos_centerness_targets.shape)
            kpt_weights = pos_centerness_targets.new_ones(len(pos_centerness_targets), 6) * self.kpt_coef
            kpt8_weights = pos_centerness_targets.new_ones(len(pos_centerness_targets), 16) * self.kpt_coef
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                assert len(code_weight) == sum(self.group_reg_dims)
                bbox_weights = bbox_weights * bbox_weights.new_tensor(
                    code_weight)

            if self.diff_rad_by_sin:
                pos_bbox_preds, pos_bbox_targets_3d = self.add_sin_difference(
                    pos_bbox_preds, pos_bbox_targets_3d)

            loss_offset = self.loss_bbox(
                pos_bbox_preds[:, :2],
                pos_bbox_targets_3d[:, :2],
                weight=bbox_weights[:, :2],
                avg_factor=equal_weights.sum())
            loss_size = self.loss_bbox(
                pos_bbox_preds[:, 3:6],
                pos_bbox_targets_3d[:, 3:6],
                weight=bbox_weights[:, 3:6],
                avg_factor=equal_weights.sum())
            loss_rotsin = self.loss_bbox(
                pos_bbox_preds[:, 6],
                pos_bbox_targets_3d[:, 6],
                weight=bbox_weights[:, 6],
                avg_factor=equal_weights.sum())
            if self.pred_velo:
                loss_dict['loss_velo'] = self.loss_bbox(
                    pos_bbox_preds[:, 7:9],
                    pos_bbox_targets_3d[:, 7:9],
                    weight=bbox_weights[:, 7:9],
                    avg_factor=equal_weights.sum())

            proj_bbox2d_inputs = (bbox_preds, pos_dir_cls_preds, labels_3d,
                                  bbox_targets_3d, pos_points, pos_inds,
                                  img_metas)

            # direction classification loss
            # TODO: add more check for use_direction_classifier
            if self.use_direction_classifier:
                loss_dict['loss_dir'] = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_cls_targets,
                    equal_weights,
                    avg_factor=equal_weights.sum())

            # if self.with_pts_depth:
            cv = pos_points[:, 1:2] - pos_bbox_propagation_preds[:, 1:2]
            spt1 = pos_points[:, 1:2] - pos_sbbox_propagation_preds[:, 1:2]
            spt2 = pos_points[:, 1:2] - pos_sbbox_propagation_preds[:, 3:4]
            spt3 = pos_points[:, 1:2] - pos_sbbox_propagation_preds[:, 5:6]
            fpt1 = pos_points[:, 1:2] - pos_fbbox_propagation_preds[:, 1:2]
            fpt2 = pos_points[:, 1:2] - pos_fbbox_propagation_preds[:, 3:4]
            fpt3 = pos_points[:, 1:2] - pos_fbbox_propagation_preds[:, 5:6]

            # init depth loss with the one computed from direct regression
            if self.with_direct_depth:
                loss_dict['loss_direct_depth'] = self.loss_depth(
                    pos_bbox_preds[:, 2],
                    pos_bbox_targets_3d[:, 2],
                    sigma=pos_weights[:, 1],
                    weight=bbox_weights[:, 2],
                    avg_factor=equal_weights.sum())

            if self.with_pts_depth and self.with_pts_uncertainty:
                loss_dict['loss_direct_spt1'] = self.loss_depth(
                    pos_sbbox_preds[:, 6], #* (spt1[:, 0] - pos_focals[:, 1]) / (cv[:, 0] - pos_focals[:, 1]) - pos_focals[:, 0] / cv[:, 0] * pos_bbox_preds[:, 4] / 2,
                    pos_bbox_targets_3d[:, 2],
                    sigma=pos_sweights[:, 0],
                    weight=bbox_weights[:, 2] * self.kpt_coef,
                    avg_factor=equal_weights.sum())
                loss_dict['loss_direct_spt2'] = self.loss_depth(
                    pos_sbbox_preds[:, 7], # * (spt2[:, 0] - pos_focals[:, 1]) / (cv[:, 0] - pos_focals[:, 1]) - pos_focals[:, 0] / cv[:, 0] * pos_bbox_preds[:, 4] / 2,
                    pos_bbox_targets_3d[:, 2],
                    sigma=pos_sweights[:, 1],
                    weight=bbox_weights[:, 2] * self.kpt_coef,
                    avg_factor=equal_weights.sum())
                loss_dict['loss_direct_spt3'] = self.loss_depth(
                    pos_sbbox_preds[:, 8], # * (spt3[:, 0] - pos_focals[:, 1]) / (cv[:, 0] - pos_focals[:, 1]) - pos_focals[:, 0] / cv[:, 0] * pos_bbox_preds[:, 4] / 2,
                    pos_bbox_targets_3d[:, 2],
                    sigma=pos_sweights[:, 2],
                    weight=bbox_weights[:, 2] * self.kpt_coef,
                    avg_factor=equal_weights.sum())

                loss_dict['loss_direct_fpt1'] = self.loss_depth(
                    pos_fbbox_preds[:, 6], # * (fpt1[:, 0] - pos_focals[:, 1]) / (cv[:, 0] - pos_focals[:, 1]) - pos_focals[:, 0] / cv[:, 0] * pos_bbox_preds[:, 4] / 2,
                    pos_bbox_targets_3d[:, 2],
                    sigma=pos_fweights[:, 0],
                    weight=bbox_weights[:, 2] * self.kpt_coef,
                    avg_factor=equal_weights.sum())
                loss_dict['loss_direct_fpt2'] = self.loss_depth(
                    pos_fbbox_preds[:, 7], # * (fpt2[:, 0] - pos_focals[:, 1]) / (cv[:, 0] - pos_focals[:, 1]) - pos_focals[:, 0] / cv[:, 0] * pos_bbox_preds[:, 4] / 2,
                    pos_bbox_targets_3d[:, 2],
                    sigma=pos_fweights[:, 1],
                    weight=bbox_weights[:, 2] * self.kpt_coef,
                    avg_factor=equal_weights.sum())
                loss_dict['loss_direct_fpt3'] = self.loss_depth(
                    pos_fbbox_preds[:, 8], # * (fpt3[:, 0] - pos_focals[:, 1]) / (cv[:, 0] - pos_focals[:, 1]) - pos_focals[:, 0] / cv[:, 0] * pos_bbox_preds[:, 4] / 2,
                    pos_bbox_targets_3d[:, 2],
                    sigma=pos_fweights[:, 2],
                    weight=bbox_weights[:, 2] * self.kpt_coef,
                    avg_factor=equal_weights.sum())
            self.with_fs_uncertainty = False
            if self.with_fs_uncertainty:
                loss_dict['loss_direct_fdepth'] = self.loss_depth(
                    pos_fbbox_preds[:, 6],
                    pos_bbox_targets_3d[:, 2],
                    sigma=pos_fweights[:, 0],
                    weight=bbox_weights[:, 2],
                    avg_factor=equal_weights.sum())
                loss_dict['loss_direct_sdepth'] = self.loss_depth(
                    pos_sbbox_preds[:, 6],
                    pos_bbox_targets_3d[:, 2],
                    sigma=pos_sweights[:, 0],
                    weight=bbox_weights[:, 2],
                    avg_factor=equal_weights.sum())
            # depth classification loss
            if self.use_depth_classifier:
                pos_prob_depth_preds = self.bbox_coder.decode_prob_depth(
                    pos_depth_cls_preds, self.depth_range, self.depth_unit,
                    self.division, self.num_depth_cls)
                sig_alpha = torch.sigmoid(self.fuse_lambda)
                if self.weight_dim != -1:
                    if self.with_pts_depth:
                        combined_depth = torch.cat([pos_bbox_preds[:, 2:3],
                                                    pos_sbbox_preds[:, 6:7], #*(spt1[:, 0:1] - pos_focals[:, 1:2]) / (cv[:, 0:1] - pos_focals[:, 1:2]) - pos_focals / cv[:, 0:1] * pos_bbox_preds[:, 4:5] / 2,
                                                    pos_sbbox_preds[:, 7:8], #*(spt2[:, 0:1] - pos_focals[:, 1:2]) / (cv[:, 0:1] - pos_focals[:, 1:2]) + pos_focals / cv[:, 0:1] * pos_bbox_preds[:, 4:5] / 2,
                                                    pos_sbbox_preds[:, 8:9], #*(spt3[:, 0:1] - pos_focals[:, 1:2]) / (cv[:, 0:1] - pos_focals[:, 1:2]) + pos_focals / cv[:, 0:1] * pos_bbox_preds[:, 4:5] / 2,
                                                    pos_fbbox_preds[:, 6:7], #*(fpt1[:, 0:1] - pos_focals[:, 1:2]) / (cv[:, 0:1] - pos_focals[:, 1:2]) - pos_focals / cv[:, 0:1] * pos_bbox_preds[:, 4:5] / 2,
                                                    pos_fbbox_preds[:, 7:8], #*(fpt2[:, 0:1] - pos_focals[:, 1:2]) / (cv[:, 0:1] - pos_focals[:, 1:2]) + pos_focals / cv[:, 0:1] * pos_bbox_preds[:, 4:5] / 2,
                                                    pos_fbbox_preds[:, 8:9], #*(fpt3[:, 0:1] - pos_focals[:, 1:2]) / (cv[:, 0:1] - pos_focals[:, 1:2]) + pos_focals / cv[:, 0:1] * pos_bbox_preds[:, 4:5] / 2,
                                                    ], dim=1)
                        combined_uncertainty = torch.cat([pos_weights[:, 1:2],
                                                          pos_sweights[:, 0:1],
                                                          pos_sweights[:, 1:2],
                                                          pos_sweights[:, 2:3],
                                                          pos_fweights[:, 0:1],
                                                          pos_fweights[:, 1:2],
                                                          pos_fweights[:, 2:3],
                                                          ], dim=1)
                    else:
                        combined_depth = torch.cat([pos_bbox_preds[:, 2:3], pos_sbbox_preds[:, 6:7], pos_fbbox_preds[:, 6:7]], dim=1)
                        combined_uncertainty = torch.cat([pos_weights[:, 1:2], pos_sweights[:, 0:1], pos_fweights[:, 0:1]], dim=1)
                    combined_weights = 1 / combined_uncertainty
                    combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
                    soft_depths = torch.sum(combined_depth * combined_weights, dim=1, keepdim=True)
                    ensemble_depth = sig_alpha * soft_depths + (1 - sig_alpha) * pos_prob_depth_preds[:, None]
                    # ensemble_depth = sig_alpha * pos_bbox_preds[:, 2:3] + (1 - sig_alpha) * pos_prob_depth_preds[:, None]
                    loss_dict['loss_ensemble_depth'] = self.loss_depth(
                        ensemble_depth[:, 0],
                        pos_bbox_targets_3d[:, 2],
                        sigma=pos_weights[:, 0],
                        weight=bbox_weights[:, 2],
                        avg_factor=equal_weights.sum())

                else:
                    pass

                proj_bbox2d_inputs += (pos_depth_cls_preds, ensemble_depth)

            if self.pred_keypoints:
                # use smoothL1 to compute consistency loss for keypoints
                # normalize the offsets with strides
                proj_bbox2d_preds, pos_decoded_bbox2d_preds, kpts_preds, kpts_targets, skpts_targets, fkpts_targets, \
                pos_ht_cls_targets, spts_yaw_targets, fpts_yaw_targets, cam_paras = \
                    self.get_proj_bbox2d(*proj_bbox2d_inputs, with_kpts=True, with_fs_kpts=True)
                # loss_dict['loss_8kpts'] = self.loss_bbox(
                #     kpts_preds,
                #     kpts_targets,
                #     weight=kpt8_weights,
                #     avg_factor=equal_weights.sum())

                loss_dict['loss_skpts'] = self.loss_bbox(
                    pos_sbbox_preds[:, :6],
                    skpts_targets,
                    weight=kpt_weights[:, :6],
                    avg_factor=equal_weights.sum())
                loss_dict['loss_fkpts'] = self.loss_bbox(
                    pos_fbbox_preds[:, :6],
                    fkpts_targets,
                    weight=kpt_weights[:, :6],
                    avg_factor=equal_weights.sum())

                # if self.with_pts_depth is None:
                if self.with_ht and self.with_fs_yaw:
                    loss_dict['loss_sht'] = self.loss_dir(
                        pos_sbbox_preds[:, -5:-3],
                        pos_ht_cls_targets,
                        equal_weights,
                        avg_factor=equal_weights.sum())
                    loss_dict['loss_fht'] = self.loss_dir(
                        pos_fbbox_preds[:, -5:-3],
                        pos_ht_cls_targets,
                        equal_weights,
                        avg_factor=equal_weights.sum())
                if self.with_ht and not self.with_fs_yaw:
                    loss_dict['loss_sht'] = self.loss_dir(
                        pos_sbbox_preds[:, -2:],
                        pos_ht_cls_targets,
                        equal_weights,
                        avg_factor=equal_weights.sum())
                    loss_dict['loss_fht'] = self.loss_dir(
                        pos_fbbox_preds[:, -2:],
                        pos_ht_cls_targets,
                        equal_weights,
                        avg_factor=equal_weights.sum())
                if self.with_fs_yaw:
                    loss_dict['loss_syaw'] = self.loss_bbox(
                        pos_sbbox_preds[:, -3:],
                        spts_yaw_targets[:, :],
                        weight=kpt_weights[:, :3],
                        avg_factor=equal_weights.sum())
                    loss_dict['loss_fyaw'] = self.loss_bbox(
                        pos_fbbox_preds[:, -3:],
                        fpts_yaw_targets[:, :],
                        weight=kpt_weights[:, -3:],
                        avg_factor=equal_weights.sum())
                    # rot constraints
                    loss_sry1 = self.loss_bbox(
                        pos_sbbox_preds[:, -3] + torch.atan2(spt1[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        pos_bbox_preds[:, 6] + torch.atan2(cv[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        weight=kpt_weights[:, 0] / 6,
                        avg_factor=equal_weights.sum())
                    loss_sry2 = self.loss_bbox(
                        pos_sbbox_preds[:, -2] + torch.atan2(spt2[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        pos_bbox_preds[:, 6] + torch.atan2(cv[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        weight=kpt_weights[:, 1] / 6,
                        avg_factor=equal_weights.sum())
                    loss_sry3 = self.loss_bbox(
                        pos_sbbox_preds[:, -1] + torch.atan2(spt3[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        pos_bbox_preds[:, 6] + torch.atan2(cv[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        weight=kpt_weights[:, 2] / 6,
                        avg_factor=equal_weights.sum())
                    loss_fry1 = self.loss_bbox(
                        pos_fbbox_preds[:, -3] + torch.atan2(fpt1[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        pos_bbox_preds[:, 6] + torch.atan2(cv[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        weight=kpt_weights[:, 3] / 6,
                        avg_factor=equal_weights.sum())
                    loss_fry2 = self.loss_bbox(
                        pos_fbbox_preds[:, -2] + torch.atan2(fpt2[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        pos_bbox_preds[:, 6] + torch.atan2(cv[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        weight=kpt_weights[:, 4] / 6,
                        avg_factor=equal_weights.sum())
                    loss_fry3 = self.loss_bbox(
                        pos_fbbox_preds[:, -1] + torch.atan2(fpt3[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        pos_bbox_preds[:, 6] + torch.atan2(cv[:, 0] - cam_paras[:, 1], cam_paras[:, 0]),
                        weight=kpt_weights[:, 5] / 6,
                        avg_factor=equal_weights.sum())
                    loss_dict['loss_rot_constraints'] = loss_sry1 + loss_sry2 + loss_sry3 + loss_fry1 + loss_fry2 + loss_fry3

            if self.pred_bbox2d:
                loss_dict['loss_bbox2d'] = self.loss_bbox2d(
                    pos_bbox_preds[:, -4:],
                    pos_bbox_targets_3d[:, -4:],
                    weight=bbox_weights[:, -4:],
                    avg_factor=equal_weights.sum())
                if not self.pred_keypoints:
                    proj_bbox2d_preds, pos_decoded_bbox2d_preds = \
                        self.get_proj_bbox2d(*proj_bbox2d_inputs)
                loss_dict['loss_consistency'] = self.loss_consistency(
                    proj_bbox2d_preds,
                    pos_decoded_bbox2d_preds,
                    weight=bbox_weights[:, -4:],
                    avg_factor=equal_weights.sum())

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)

            # attribute classification loss
            if self.pred_attrs:
                loss_dict['loss_attr'] = self.loss_attr(
                    pos_attr_preds,
                    pos_attr_targets,
                    pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())

        else:
            # need absolute due to possible negative delta x/y
            loss_offset = pos_bbox_preds[:, :2].sum()
            loss_size = pos_bbox_preds[:, 3:6].sum()
            loss_rotsin = pos_bbox_preds[:, 6].sum()
            loss_dict['loss_depth'] = pos_bbox_preds[:, 2].sum()
            if self.pred_velo:
                loss_dict['loss_velo'] = pos_bbox_preds[:, 7:9].sum()
            if self.pred_keypoints:
                loss_dict['loss_skpts'] = pos_sbbox_preds[:, :6].sum()
                loss_dict['loss_fkpts'] = pos_fbbox_preds[:, :6].sum()
            if self.pred_bbox2d:
                loss_dict['loss_bbox2d'] = pos_bbox_preds[:, -4:].sum()
                loss_dict['loss_consistency'] = pos_bbox_preds[:, -4:].sum()
            loss_centerness = pos_centerness.sum()
            if self.use_direction_classifier:
                loss_dict['loss_dir'] = pos_dir_cls_preds.sum()
            if self.use_depth_classifier:
                if self.weight_dim != -1:
                    loss_fuse_depth = \
                        pos_bbox_preds[:, 2].sum() + \
                        pos_depth_cls_preds.sum()
                    loss_fuse_depth *= torch.exp(-pos_weights[:, 0].sum())
                else:
                    loss_fuse_depth = \
                        pos_bbox_preds[:, 2].sum() + \
                        pos_depth_cls_preds.sum()
                loss_dict['loss_depth'] = loss_fuse_depth
            if self.pred_attrs:
                loss_dict['loss_attr'] = pos_attr_preds.sum()

        loss_dict.update(
            dict(
                loss_cls=loss_cls,
                loss_offset=loss_offset,
                loss_size=loss_size,
                loss_rotsin=loss_rotsin,
                loss_centerness=loss_centerness))

        return loss_dict

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds',
                  'depth_cls_preds', 'weights', 'attr_preds', 'centernesses',
                  'sbbox_preds', 'sweights', 'fbbox_preds', 'fweights'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   depth_cls_preds,
                   weights,
                   attr_preds,
                   centernesses,
                   sbbox_preds,
                   sweights,
                   fbbox_preds,
                   fweights,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            depth_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * self.num_depth_cls.
            weights (list[Tensor]): Location-aware weights for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * self.weight_dim.
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config, optional): Test / postprocessing configuration,
                if None, test_cfg would be used. Defaults to None.
            rescale (bool, optional): If True, return boxes in original image
                space. Defaults to None.

        Returns:
            list[tuple[Tensor]]: Each item in result_list is a tuple, which
                consists of predicted 3D boxes, scores, labels, attributes and
                2D boxes (if necessary).
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == \
            len(depth_cls_preds) == len(weights) == len(centernesses) == \
            len(attr_preds), 'The length of cls_scores, bbox_preds, ' \
            'dir_cls_preds, depth_cls_preds, weights, centernesses, and' \
            f'attr_preds: {len(cls_scores)}, {len(bbox_preds)}, ' \
            f'{len(dir_cls_preds)}, {len(depth_cls_preds)}, {len(weights)}' \
            f'{len(centernesses)}, {len(attr_preds)} are inconsistent.'
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            sbbox_pred_list = [
                sbbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            fbbox_pred_list = [
                fbbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if self.use_direction_classifier:
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [2, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]
            if self.use_depth_classifier:
                depth_cls_pred_list = [
                    depth_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                depth_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_depth_cls, *cls_scores[i][img_id].shape[1:]],
                        0).detach() for i in range(num_levels)
                ]
            if self.weight_dim != -1:
                weight_list = [
                    weights[i][img_id].detach() for i in range(num_levels)
                ]
                sweight_list = [
                    sweights[i][img_id].detach() for i in range(num_levels)
                ]
                fweight_list = [
                    fweights[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                weight_list = [
                    cls_scores[i][img_id].new_full(
                        [1, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]
            if self.pred_attrs:
                attr_pred_list = [
                    attr_preds[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                attr_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [self.num_attrs, *cls_scores[i][img_id].shape[1:]],
                        self.attr_background_label).detach()
                    for i in range(num_levels)
                ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            input_meta = img_metas[img_id]
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                depth_cls_pred_list, weight_list, attr_pred_list,
                centerness_pred_list, sbbox_pred_list, sweight_list,
                fbbox_pred_list, fweight_list, mlvl_points, input_meta, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           depth_cls_preds,
                           weights,
                           attr_preds,
                           centernesses,
                           sbbox_preds,
                           sweights,
                           fbbox_preds,
                           fweights,
                           mlvl_points,
                           input_meta,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * bbox_code_size, H, W).
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on a single scale level with shape
                (num_points * 2, H, W)
            depth_cls_preds (list[Tensor]): Box scores for probabilistic depth
                predictions on a single scale level with shape
                (num_points * self.num_depth_cls, H, W)
            weights (list[Tensor]): Location-aware weight maps on a single
                scale level with shape (num_points * self.weight_dim, H, W).
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            input_meta (dict): Metadata of input image.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool, optional): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            tuples[Tensor]: Predicted 3D boxes, scores, labels, attributes and
                2D boxes (if necessary).
        """
        view = np.array(input_meta['cam2img'])
        scale_factor = input_meta['scale_factor']
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_sht_scores = []
        mlvl_fht_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []
        mlvl_depth_cls_scores = []
        mlvl_depth_uncertainty = []
        mlvl_direct_uncertainty = []
        mlvl_bboxes2d = None
        if self.pred_bbox2d:
            mlvl_bboxes2d = []


        for cls_score, bbox_pred, dir_cls_pred, depth_cls_pred, weight, \
                attr_pred, centerness, sbbox_pred, sweight, fbbox_pred, fweight, points\
                in zip(cls_scores, bbox_preds, dir_cls_preds, depth_cls_preds,
                    weights, attr_preds, centernesses, sbbox_preds, sweights, fbbox_preds, fweights, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            depth_cls_pred = depth_cls_pred.permute(1, 2, 0).reshape(-1, self.num_depth_cls)
            depth_cls_score = F.softmax(depth_cls_pred, dim=-1).topk(k=2, dim=-1)[0].mean(dim=-1)
            if self.weight_dim != -1:
                weight = weight.permute(1, 2, 0).reshape(-1, self.weight_dim)
                if self.with_pts_depth:
                    sweight = sweight.permute(1, 2, 0).reshape(-1, 3)
                    fweight = fweight.permute(1, 2, 0).reshape(-1, 3)
                else:
                    sweight = sweight.permute(1, 2, 0).reshape(-1, 1)
                    fweight = fweight.permute(1, 2, 0).reshape(-1, 1)
            else:
                weight = weight.permute(1, 2, 0).reshape(-1, 1)
            depth_uncertainty = torch.exp(-weight[:, 0])
            # add direct-depth-regression uncertainty and fs-depth-regression uncertainty
            # direct_uncertainty = torch.exp(-weight[:, 0])
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, sum(self.group_reg_dims))
            if self.with_pts_depth:
                self.pts_dims = 9
                if self.with_ht:
                    self.pts_dims += 2
                if self.with_fs_yaw:
                    self.pts_dims += 3
                sbbox_pred = sbbox_pred.permute(1, 2, 0).reshape(-1, self.pts_dims)
                fbbox_pred = fbbox_pred.permute(1, 2, 0).reshape(-1, self.pts_dims)
            else:
                self.fs_dims = 7
                if self.with_ht:
                    self.fs_dims += 2
                if self.with_fs_yaw:
                    self.fs_dims += 3
                sbbox_pred = sbbox_pred.permute(1, 2, 0).reshape(-1, self.fs_dims)
                fbbox_pred = fbbox_pred.permute(1, 2, 0).reshape(-1, self.fs_dims)
            if self.with_ht:
                sht_cls_pred = sbbox_pred[:, 7:9].reshape(-1, 2)
                sht_cls_score = torch.max(sht_cls_pred, dim=-1)[1]
                fht_cls_pred = fbbox_pred[:, 7:9].reshape(-1, 2)
                fht_cls_score = torch.max(fht_cls_pred, dim=-1)[1]
            bbox_pred3d = bbox_pred[:, :self.bbox_coder.bbox_code_size]
            if self.pred_bbox2d:
                bbox_pred2d = bbox_pred[:, -4:]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                merged_scores = scores * centerness[:, None]
                if self.use_depth_classifier:
                    merged_scores *= depth_cls_score[:, None]
                    if self.weight_dim != -1:
                        merged_scores *= depth_uncertainty[:, None]
                max_scores, _ = merged_scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred3d = bbox_pred3d[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                depth_cls_pred = depth_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                if self.with_ht:
                    sht_cls_score = sht_cls_score[topk_inds]
                    fht_cls_score = fht_cls_score[topk_inds]
                depth_cls_score = depth_cls_score[topk_inds]
                depth_uncertainty = depth_uncertainty[topk_inds]
                # direct_uncertainty = direct_uncertainty[topk_inds]
                attr_score = attr_score[topk_inds]
                bbox_pred = bbox_pred[topk_inds, :]
                weight = weight[topk_inds, :]
                sbbox_pred = sbbox_pred[topk_inds, :]
                sweight = sweight[topk_inds, :]
                fbbox_pred = fbbox_pred[topk_inds, :]
                fweight = fweight[topk_inds, :]
                if self.pred_bbox2d:
                    bbox_pred2d = bbox_pred2d[topk_inds, :]
            # change the offset to actual center predictions
            bbox_pred3d[:, :2] = points - bbox_pred3d[:, :2]
            if rescale:
                bbox_pred3d[:, :2] /= bbox_pred3d[:, :2].new_tensor(
                    scale_factor)
                if self.pred_bbox2d:
                    bbox_pred2d /= bbox_pred2d.new_tensor(scale_factor)
            if self.use_depth_classifier:
                prob_depth_pred = self.bbox_coder.decode_prob_depth(
                    depth_cls_pred, self.depth_range, self.depth_unit,
                    self.division, self.num_depth_cls)
                sig_alpha = torch.sigmoid(self.fuse_lambda)
                # bbox_pred3d[:, 2] = sig_alpha * bbox_pred3d[:, 2] + (1 - sig_alpha) * prob_depth_pred
                if self.with_pts_depth:
                    cv = points[:, 1:2] - bbox_pred[:, 1:2]
                    spt1 = points[:, 1:2] - sbbox_pred[:, 1:2]
                    spt2 = points[:, 1:2] - sbbox_pred[:, 3:4]
                    spt3 = points[:, 1:2] - sbbox_pred[:, 5:6]
                    fpt1 = points[:, 1:2] - fbbox_pred[:, 1:2]
                    fpt2 = points[:, 1:2] - fbbox_pred[:, 3:4]
                    fpt3 = points[:, 1:2] - fbbox_pred[:, 5:6]
                    combined_depth = torch.cat([bbox_pred[:, 2:3],
                                                sbbox_pred[:, 6:7], # * spt1[:, 0:1] / cv[:, 0:1] - view[0][0]/cv[:, 0:1]*bbox_pred[:, 4:5]/2,
                                                sbbox_pred[:, 7:8], # * spt2[:, 0:1] / cv[:, 0:1] + view[0][0]/cv[:, 0:1]*bbox_pred[:, 4:5]/2,
                                                sbbox_pred[:, 8:9], # * spt3[:, 0:1] / cv[:, 0:1] + view[0][0]/cv[:, 0:1]*bbox_pred[:, 4:5]/2,
                                                fbbox_pred[:, 6:7], # * fpt1[:, 0:1] / cv[:, 0:1] - view[0][0]/cv[:, 0:1]*bbox_pred[:, 4:5]/2,
                                                fbbox_pred[:, 7:8], # * fpt2[:, 0:1] / cv[:, 0:1] + view[0][0]/cv[:, 0:1]*bbox_pred[:, 4:5]/2,
                                                fbbox_pred[:, 8:9], # * fpt3[:, 0:1] / cv[:, 0:1] + view[0][0]/cv[:, 0:1]*bbox_pred[:, 4:5]/2,
                                                ], dim=1)
                    combined_uncertainty = torch.cat([weight[:, 1:2],
                                                      sweight[:, 0:1],
                                                      sweight[:, 1:2],
                                                      sweight[:, 2:3],
                                                      fweight[:, 0:1],
                                                      fweight[:, 1:2],
                                                      fweight[:, 2:3],
                                                      ], dim=1)
                else:
                    combined_depth = torch.cat([bbox_pred3d[:, 2:3], sbbox_pred[:, 6:7], fbbox_pred[:, 6:7]], dim=1)
                    combined_uncertainty = torch.cat([weight[:, 1:2], sweight[:, 0:1], fweight[:, 0:1]], dim=1)
                combined_weights = 1 / combined_uncertainty
                combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
                soft_depths = torch.sum(combined_depth * combined_weights, dim=1, keepdim=True)
                ensemble_depth = sig_alpha * soft_depths + (1 - sig_alpha) * prob_depth_pred[:, None]
                # ensemble_depth = sig_alpha * bbox_pred[:, 2:3] + (1 - sig_alpha) * prob_depth_pred[:, None]
                bbox_pred3d[:, 2] = ensemble_depth[:, 0]

            pred_center2d = bbox_pred3d[:, :3].clone()
            bbox_pred3d[:, :3] = points_img2cam(bbox_pred3d[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred3d)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            if self.with_ht:
                mlvl_sht_scores.append(sht_cls_score)
                mlvl_fht_scores.append(fht_cls_score)
            mlvl_depth_cls_scores.append(depth_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)
            mlvl_depth_uncertainty.append(depth_uncertainty)
            # mlvl_direct_uncertainty.append(direct_uncertainty)
            if self.pred_bbox2d:
                bbox_pred2d = distance2bbox(
                    points, bbox_pred2d, max_shape=input_meta['img_shape'])
                mlvl_bboxes2d.append(bbox_pred2d)

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        if self.with_ht:
            mlvl_sht_scores = torch.cat(mlvl_sht_scores)
            mlvl_fht_scores = torch.cat(mlvl_fht_scores)
        if self.pred_bbox2d:
            mlvl_bboxes2d = torch.cat(mlvl_bboxes2d)

        # change local yaw to global yaw for 3D nms
        cam2img = torch.eye(
            4, dtype=mlvl_centers2d.dtype, device=mlvl_centers2d.device)
        cam2img[:view.shape[0], :view.shape[1]] = \
            mlvl_centers2d.new_tensor(view)
        if self.with_ht:
            mlvl_bboxes = self.bbox_coder.decode_ht(mlvl_bboxes, mlvl_centers2d,
                                                    mlvl_dir_scores, mlvl_sht_scores,
                                                    self.dir_offset, cam2img)
        else:
            mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                                    mlvl_dir_scores,
                                                    self.dir_offset, cam2img)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        # no scale_factors in box3d_multiclass_nms
        # Then we multiply it from outside
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        if self.use_depth_classifier:  # multiply the depth confidence
            mlvl_depth_cls_scores = torch.cat(mlvl_depth_cls_scores)
            mlvl_nms_scores *= mlvl_depth_cls_scores[:, None]
            if self.weight_dim != -1:
                mlvl_depth_uncertainty = torch.cat(mlvl_depth_uncertainty)
                # mlvl_direct_uncertainty = torch.cat(mlvl_direct_uncertainty)
                mlvl_nms_scores *= mlvl_depth_uncertainty[:, None]
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg.score_thr,
                                       cfg.max_per_img, cfg, mlvl_dir_scores,
                                       mlvl_attr_scores, mlvl_bboxes2d)
        bboxes, scores, labels, dir_scores, attrs = results[0:5]
        attrs = attrs.to(labels.dtype)  # change data type to int
        bboxes = input_meta['box_type_3d'](
            bboxes,
            box_dim=self.bbox_coder.bbox_code_size,
            origin=(0.5, 0.5, 0.5))
        # Note that the predictions use origin (0.5, 0.5, 0.5)
        # Due to the ground truth centers2d are the gravity center of objects
        # v0.10.0 fix inplace operation to the input tensor of cam_box3d
        # So here we also need to add origin=(0.5, 0.5, 0.5)
        if not self.pred_attrs:
            attrs = None

        outputs = (bboxes, scores, labels, attrs)
        if self.pred_bbox2d:
            bboxes2d = results[-1]
            bboxes2d = torch.cat([bboxes2d, scores[:, None]], dim=1)
            outputs = outputs + (bboxes2d, )

        return outputs

    @staticmethod
    def get_hs_target(reg_targets, dir_offset=0, dir_limit_offset=0.0, num_bins=2, one_hot=True):
        """Encode direction to 0 ~ num_bins-1.

        Args:
            reg_targets (torch.Tensor): Bbox regression targets.
            dir_offset (int, optional): Direction offset. Default to 0.
            dir_limit_offset (float, optional): Offset to set the direction
                range. Default to 0.0.
            num_bins (int, optional): Number of bins to divide 2*PI.
                Default to 2.
            one_hot (bool, optional): Whether to encode as one hot.
                Default to True.

        Returns:
            torch.Tensor: Encoded direction targets.
        """
        rot_gt = reg_targets[..., 6] + reg_targets[...]
        offset_rot = limit_period(rot_gt - dir_offset, dir_limit_offset,
                                  2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot /
                                      (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(
                *list(dir_cls_targets.shape),
                num_bins,
                dtype=reg_targets.dtype,
                device=dir_cls_targets.device)
            dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_targets(self, points, gt_bboxes_list, gt_labels_list,
                    gt_bboxes_3d_list, gt_labels_3d_list, centers2d_list,
                    depths_list, attr_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            gt_bboxes_3d_list (list[Tensor]): 3D Ground truth bboxes of each
                image, each has shape (num_gt, bbox_code_size).
            gt_labels_3d_list (list[Tensor]): 3D Ground truth labels of each
                box, each has shape (num_gt,).
            centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
                each has shape (num_gt, 2).
            depths_list (list[Tensor]): Depth of projected 3D centers onto 2D
                image, each has shape (num_gt, 1).
            attr_labels_list (list[Tensor]): Attribute labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        expanded_depth_ranges = [
            points[i].new_tensor(self.depth_layer_ranges[i])[None].expand_as(points[i])
            for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_depth_ranges = torch.cat(expanded_depth_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        if attr_labels_list is None:
            attr_labels_list = [
                gt_labels.new_full(gt_labels.shape, self.attr_background_label)
                for gt_labels in gt_labels_list
            ]

        # get labels and bbox_targets of each image
        _, bbox_targets_list, labels_3d_list, bbox_targets_3d_list, \
            centerness_targets_list, attr_targets_list = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_3d_list,
                gt_labels_3d_list,
                centers2d_list,
                depths_list,
                attr_labels_list,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                depth_ranges=concat_depth_ranges,
                num_points_per_lvl=num_points)

        # split to per img, per level
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        labels_3d_list = [
            labels_3d.split(num_points, 0) for labels_3d in labels_3d_list
        ]
        bbox_targets_3d_list = [
            bbox_targets_3d.split(num_points, 0)
            for bbox_targets_3d in bbox_targets_3d_list
        ]
        centerness_targets_list = [
            centerness_targets.split(num_points, 0)
            for centerness_targets in centerness_targets_list
        ]
        attr_targets_list = [
            attr_targets.split(num_points, 0)
            for attr_targets in attr_targets_list
        ]

        # concat per level image
        concat_lvl_labels_3d = []
        concat_lvl_bbox_targets_3d = []
        concat_lvl_centerness_targets = []
        concat_lvl_attr_targets = []
        for i in range(num_levels):
            concat_lvl_labels_3d.append(
                torch.cat([labels[i] for labels in labels_3d_list]))
            concat_lvl_centerness_targets.append(
                torch.cat([
                    centerness_targets[i]
                    for centerness_targets in centerness_targets_list
                ]))
            bbox_targets_3d = torch.cat([
                bbox_targets_3d[i] for bbox_targets_3d in bbox_targets_3d_list
            ])
            if self.pred_bbox2d:
                bbox_targets = torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list])
                bbox_targets_3d = torch.cat([bbox_targets_3d, bbox_targets],
                                            dim=1)
            concat_lvl_attr_targets.append(
                torch.cat(
                    [attr_targets[i] for attr_targets in attr_targets_list]))
            if self.norm_on_bbox:
                bbox_targets_3d[:, :2] = \
                    bbox_targets_3d[:, :2] / self.strides[i]
                if self.pred_bbox2d:
                    bbox_targets_3d[:, -4:] = \
                        bbox_targets_3d[:, -4:] / self.strides[i]
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)
        return concat_lvl_labels_3d, concat_lvl_bbox_targets_3d, \
            concat_lvl_centerness_targets, concat_lvl_attr_targets

    def _get_target_single(self, gt_bboxes, gt_labels, gt_bboxes_3d,
                           gt_labels_3d, centers2d, depths, attr_labels,
                           points, regress_ranges, depth_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if not isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels_3d.new_full(
                       (num_points,), self.background_label), \
                   gt_bboxes_3d.new_zeros((num_points, self.bbox_code_size)), \
                   gt_bboxes_3d.new_zeros((num_points,)), \
                   attr_labels.new_full(
                       (num_points,), self.attr_background_label)

        # change orientation to local yaw
        gt_bboxes_3d[..., 6] = -torch.atan2(
            gt_bboxes_3d[..., 0], gt_bboxes_3d[..., 2]) + gt_bboxes_3d[..., 6]

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        depth_ranges = depth_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers2d = centers2d[None].expand(num_points, num_gts, 2)
        gt_bboxes_3d = gt_bboxes_3d[None].expand(num_points, num_gts,
                                                 self.bbox_code_size)
        depths = depths[None, :, None].expand(num_points, num_gts, 1)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        delta_xs = (xs - centers2d[..., 0])[..., None]
        delta_ys = (ys - centers2d[..., 1])[..., None]
        bbox_targets_3d = torch.cat(
            (delta_xs, delta_ys, depths, gt_bboxes_3d[..., 3:]), dim=-1)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        assert self.center_sampling is True, 'Setting center_sampling to '\
            'False has not been implemented for FCOS3D.'
        # condition1: inside a `center bbox`
        radius = self.fsg_center_sample_radius
        center_xs = centers2d[..., 0]
        center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))
        inside_depth_range = (
            (depths[..., 0] >= depth_ranges[..., 0])
            & (depths[..., 0] <= depth_ranges[..., 1])
        )
        # center-based criterion to deal with ambiguity
        dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2]**2, dim=-1))
        dists[inside_gt_bbox_mask == 0] = 1e8
        # dists[inside_regress_range == 0] = 1e8
        dists[inside_depth_range == 0] = 1e8
        min_dist, min_dist_inds = dists.min(dim=1)

        labels = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds]
        attr_labels = attr_labels[min_dist_inds]
        labels[min_dist == 1e8] = self.background_label  # set as BG
        labels_3d[min_dist == 1e8] = self.background_label  # set as BG
        attr_labels[min_dist == 1e8] = self.attr_background_label

        bbox_targets = bbox_targets[range(num_points), min_dist_inds]
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds]
        relative_dists = torch.sqrt(
            torch.sum(bbox_targets_3d[..., :2]**2,
                      dim=-1)) / (1.414 * stride[:, 0])
        # [N, 1] / [N, 1]
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists)

        return labels, bbox_targets, labels_3d, bbox_targets_3d, \
            centerness_targets, attr_labels