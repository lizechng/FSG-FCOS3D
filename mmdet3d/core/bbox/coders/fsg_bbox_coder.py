import numpy as np
import torch
from torch.nn import functional as F

from mmdet.core.bbox.builder import BBOX_CODERS
from .fcos3d_bbox_coder import FCOS3DBBoxCoder
from ..structures import limit_period

@BBOX_CODERS.register_module()
class FSGBBoxCoder(FCOS3DBBoxCoder):
    """Bounding box coder for PGD."""

    def encode(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels):
        # TODO: refactor the encoder codes in the FCOS3D and PGD head
        pass

    def decode(self, bbox, scale, stride, training, depth_layer_base=None, cls_score=None, with_size_prior=True):
        """Decode regressed results into 3D predictions.

        Note that offsets are not transformed to the projected 3D centers.

        Args:
            bbox (torch.Tensor): Raw bounding box predictions in shape
                [N, C, H, W].
            scale (tuple[`Scale`]): Learnable scale parameters.
            stride (int): Stride for a specific feature level.
            training (bool): Whether the decoding is in the training
                procedure.
            cls_score (torch.Tensor): Classification score map for deciding
                which base depth or dim is used. Defaults to None.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        # scale the bbox of different level
        scale_offset, scale_depth, scale_size = scale[0:3]

        clone_bbox = bbox.clone()
        bbox[:, :2] = scale_offset(clone_bbox[:, :2]).float()
        bbox[:, 2] = scale_depth(clone_bbox[:, 2]).float()
        bbox[:, 3:6] = scale_size(clone_bbox[:, 3:6]).float()

        if depth_layer_base is not None:
            # bbox[:, 2] = (bbox.clone()[:, 2] / depth_layer_base).exp() # .log2()
            bbox[:, 2] = (bbox.clone()[:, 2] * depth_layer_base).exp() # .log2()
        else:
            if self.base_depths is None:
                bbox[:, 2] = bbox[:, 2].exp()
            elif len(self.base_depths) == 1:  # only single prior
                mean = self.base_depths[0][0]
                std = self.base_depths[0][1]
                bbox[:, 2] = mean + bbox.clone()[:, 2] * std
            else:  # multi-class priors
                assert len(self.base_depths) == cls_score.shape[1], \
                    'The number of multi-class depth priors should be equal to ' \
                    'the number of categories.'
                indices = cls_score.max(dim=1)[1]
                depth_priors = cls_score.new_tensor(
                    self.base_depths)[indices, :].permute(0, 3, 1, 2)
                mean = depth_priors[:, 0]
                std = depth_priors[:, 1]
                bbox[:, 2] = mean + bbox.clone()[:, 2] * std

        bbox[:, 3:6] = bbox[:, 3:6].exp()
        if self.base_dims is not None and with_size_prior:
            assert len(self.base_dims) == cls_score.shape[1], \
                'The number of anchor sizes should be equal to the number ' \
                'of categories.'
            indices = cls_score.max(dim=1)[1]
            size_priors = cls_score.new_tensor(
                self.base_dims)[indices, :].permute(0, 3, 1, 2)
            bbox[:, 3:6] = size_priors * bbox.clone()[:, 3:6]

        assert self.norm_on_bbox is True, 'Setting norm_on_bbox to False '\
            'has not been thoroughly tested for FCOS3D.'
        if self.norm_on_bbox:
            if not training:
                # Note that this line is conducted only when testing
                bbox[:, :2] *= stride

        return bbox


    def decode_2d(self,
                  bbox,
                  scale,
                  stride,
                  training,
                  pred_keypoints=False,
                  pred_bbox2d=True):
        """Decode regressed 2D attributes.

        Args:
            bbox (torch.Tensor): Raw bounding box predictions in shape
                [N, C, H, W].
            scale (tuple[`Scale`]): Learnable scale parameters.
            stride (int): Stride for a specific feature level.
            max_regress_range (int): Maximum regression range for a specific
                feature level.
            training (bool): Whether the decoding is in the training
                procedure.
            pred_keypoints (bool, optional): Whether to predict keypoints.
                Defaults to False.
            pred_bbox2d (bool, optional): Whether to predict 2D bounding
                boxes. Defaults to False.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        clone_bbox = bbox.clone()
        if pred_keypoints:
            scale_kpts = scale[3]
            # 2 dimension of offsets x 8 corners of a 3D bbox
            bbox[:, self.bbox_code_size:self.bbox_code_size + 16] = \
                scale_kpts(clone_bbox[:, self.bbox_code_size:self.bbox_code_size + 16]).float()

        if pred_bbox2d:
            scale_bbox2d = scale[-1]
            # The last four dimensions are offsets to four sides of a 2D bbox
            bbox[:, -4:] = scale_bbox2d(clone_bbox[:, -4:]).float()

        if self.norm_on_bbox:
            if pred_bbox2d:
                bbox[:, -4:] = F.relu(bbox.clone()[:, -4:])
            if not training:
                if pred_keypoints:
                    # pos_strides * self.regress_ranges[0][1] / self.strides[0] is equal to self.regress_ranges[i][1]
                    bbox[:, self.bbox_code_size:self.bbox_code_size + 16] *= \
                           stride
                if pred_bbox2d:
                    bbox[:, -4:] *= stride
        else:
            if pred_bbox2d:
                bbox[:, -4:] = bbox.clone()[:, -4:].exp()
        return bbox

    def decode_prob_depth(self, depth_cls_preds, depth_range, depth_unit,
                          division, num_depth_cls):
        """Decode probabilistic depth map.

        Args:
            depth_cls_preds (torch.Tensor): Depth probabilistic map in shape
                [..., self.num_depth_cls] (raw output before softmax).
            depth_range (tuple[float]): Range of depth estimation.
            depth_unit (int): Unit of depth range division.
            division (str): Depth division method. Options include 'uniform',
                'linear', 'log', 'loguniform'.
            num_depth_cls (int): Number of depth classes.

        Returns:
            torch.Tensor: Decoded probabilistic depth estimation.
        """
        if division == 'uniform':
            depth_multiplier = depth_unit * \
                depth_cls_preds.new_tensor(
                    list(range(num_depth_cls))).reshape([1, -1])
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                depth_multiplier).sum(dim=-1)
            return prob_depth_preds
        elif division == 'linear':
            split_pts = depth_cls_preds.new_tensor(list(
                range(num_depth_cls))).reshape([1, -1])
            depth_multiplier = depth_range[0] + (
                depth_range[1] - depth_range[0]) / \
                (num_depth_cls * (num_depth_cls - 1)) * \
                (split_pts * (split_pts+1))
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                depth_multiplier).sum(dim=-1)
            return prob_depth_preds
        elif division == 'log':
            split_pts = depth_cls_preds.new_tensor(list(
                range(num_depth_cls))).reshape([1, -1])
            start = max(depth_range[0], 1)
            end = depth_range[1]
            depth_multiplier = (np.log(start) +
                                split_pts * np.log(end / start) /
                                (num_depth_cls - 1)).exp()
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                depth_multiplier).sum(dim=-1)
            return prob_depth_preds
        elif division == 'loguniform':
            split_pts = depth_cls_preds.new_tensor(list(
                range(num_depth_cls))).reshape([1, -1])
            start = max(depth_range[0], 1)
            end = depth_range[1]
            log_multiplier = np.log(start) + \
                split_pts * np.log(end / start) / (num_depth_cls - 1)
            prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                                log_multiplier).sum(dim=-1).exp()
            return prob_depth_preds
        else:
            raise NotImplementedError

    @staticmethod
    def decode_ht(bbox, centers2d, dir_cls, ht_cls, dir_offset, cam2img):
        """Decode yaw angle and change it from local to global.i.

        Args:
            bbox (torch.Tensor): Bounding box predictions in shape
                [N, C] with yaws to be decoded.
            centers2d (torch.Tensor): Projected 3D-center on the image planes
                corresponding to the box predictions.
            dir_cls (torch.Tensor): Predicted direction classes.
            dir_offset (float): Direction offset before dividing all the
                directions into several classes.
            cam2img (torch.Tensor): Camera intrinsic matrix in shape [4, 4].

        Returns:
            torch.Tensor: Bounding boxes with decoded yaws.
        """
        if bbox.shape[0] > 0:
            dir_rot = limit_period(bbox[..., 6] - dir_offset, 0, np.pi)
            bbox[..., 6] = \
                dir_rot + dir_offset + np.pi * dir_cls.to(bbox.dtype)

        bbox[:, 6] = torch.atan2(centers2d[:, 0] - cam2img[0, 2], cam2img[0, 0]) + bbox[:, 6]

        # ht_rot = limit_period(bbox[:, 6], 0, np.pi)
        # bbox[:, 6] = ht_rot + np.pi * ht_cls.to(bbox.dtype) + np.pi

        # ht_rot = limit_period(bbox[:, 6], 0, 2 * np.pi)
        # for idx in range(bbox.shape[0]):
        #     if ht_rot[idx] < np.pi and ht_cls[idx] == 1:
        #         bbox[idx, 6] = ht_rot[idx] + np.pi
        #     elif ht_rot[idx] > np.pi and ht_cls[idx] == 0:
        #         bbox[idx, 6] = ht_rot[idx] - np.pi

        return bbox
