import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
import torch
from .builder import DATASETS
from torch.utils.data import Dataset
from .pipelines import Compose
import copy
from tqdm import tqdm
import math

from ..core.bbox import Box3DMode, CameraInstance3DBoxes, points_cam2img, get_box_type


@DATASETS.register_module()
class Mono3d(Dataset):
    CLASSES = ['Car']

    def __init__(self,
                 img_dir,
                 calib_dir,
                 idx_file,
                 pipeline,
                 info_file=None,
                 box_type_3d='Camera',
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 classes=None,
                 test_mode=False,
                 ann_dir=None,
                 filter_empty_gt=True):
        self.img_dir = img_dir
        self.calib_dir = calib_dir
        self.idx_file = idx_file
        self.test_mode = test_mode
        self.ann_dir = ann_dir
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.pipeline = Compose(pipeline)
        self.sampler_num = 0
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        self.cls2id = dict(Car=0, Cyclist=1, Pedestrian=2, Van=3)
        self.bbox_code_size = 7
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.img_infos = self.load_imgs(idx_file, img_dir, calib_dir)
        self.ann_infos = self.load_anns(idx_file, ann_dir, calib_dir)
        if self.proposal_file is not None:
            raise NotImplementedError
        else:
            self.proposals = None
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            self.ann_infos = [self.ann_infos[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()
        if info_file is not None:
            self.anno_infos = mmcv.load(info_file)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _parse_ann_info(self, img_info, ann_info):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        gt_bboxes_scam3d, gt_bboxes_fcam3d = [], []
        scenters2d, fcenters2d = [], []
        sdepths, fdepths = [], []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            assert ann['image_id'] == img_info['id']
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1+w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1+h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0 or inter_w == img_info['width'] or inter_h == img_info['height']:
                continue
            if w < 1 or h < 1:
                continue
            bbox = [max(x1, 0), max(y1, 0), min(x1 + w, img_info['width']), min(y1 + h, img_info['height'])]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(ann['category_id'])
                gt_masks_ann.append(ann.get('segmentation', None))
                # 3D annotations in camera coordinates
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(-1,)
                bbox_scam3d = np.array(ann['bbox_scam3d']).reshape(-1,)
                bbox_fcam3d = np.array(ann['bbox_fcam3d']).reshape(-1,)
                gt_bboxes_cam3d.append(bbox_cam3d)
                gt_bboxes_scam3d.append(bbox_scam3d)
                gt_bboxes_fcam3d.append(bbox_fcam3d)
                # 2.5D annotations in camera coordinates
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                scenter2d = ann['scenter2d'][:2]
                sdepth = ann['scenter2d'][2]
                fcenter2d = ann['fcenter2d'][:2]
                fdepth = ann['fcenter2d'][2]
                centers2d.append(center2d)
                depths.append(depth)
                scenters2d.append(scenter2d)
                sdepths.append(sdepth)
                fcenters2d.append(fcenter2d)
                fdepths.append(fdepth)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
            # side
            gt_bboxes_scam3d = np.array(gt_bboxes_scam3d, dtype=np.float32)
            scenters2d = np.array(scenters2d, dtype=np.float32)
            sdepths = np.array(sdepths, dtype=np.float32)
            # front
            gt_bboxes_fcam3d = np.array(gt_bboxes_fcam3d, dtype=np.float32)
            fcenters2d = np.array(fcenters2d, dtype=np.float32)
            fdepths = np.array(fdepths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size), dtype=np.float32)
            centers2d = np.array((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)
            # side
            gt_bboxes_scam3d = np.zeros((0, self.bbox_code_size), dtype=np.float32)
            scenters2d = np.array((0, 2), dtype=np.float32)
            sdepths = np.zeros((0), dtype=np.float32)
            # front
            gt_bboxes_fcam3d = np.zeros((0, self.bbox_code_size), dtype=np.float32)
            fcenters2d = np.array((0, 2), dtype=np.float32)
            fdepths = np.zeros((0), dtype=np.float32)
        # The influence of 'CameraInstance3DBoxes' in training phase.
        # As shown in 'PGD-FCOS3D', the class 'CameraInstance3DBoxes' is for image flip
        # That paper implement image flip for augmentation, where offset and 2D targets are
        # flipped for the 2D image while 3D boxes are transformed correspondingly in 3D space.
        gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d,
            box_dim=gt_bboxes_cam3d.shape[-1],
            origin=(0.5, 1.0, 0.5)
        )
        # side & front 'CameraInstance3DBoxes' only tested for flip()
        gt_bboxes_scam3d = CameraInstance3DBoxes(
            gt_bboxes_scam3d,
            box_dim=gt_bboxes_scam3d.shape[-1],
            origin=(0.5, 0.5, 0.5)
        )
        gt_bboxes_fcam3d = CameraInstance3DBoxes(
            gt_bboxes_fcam3d,
            box_dim=gt_bboxes_fcam3d.shape[-1],
            origin=(0.5, 0.5, 0.5)
        )
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d,
            gt_bboxes_s3d=gt_bboxes_scam3d, # side
            gt_bboxes_f3d=gt_bboxes_fcam3d, # front
            gt_labels_3d=gt_labels_3d,
            centers2d=centers2d,
            scenters2d=scenters2d,
            fcenters2d=fcenters2d,
            depths=depths,
            sdepths=sdepths,
            fdepths=fdepths,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
        )

        return ann

    def get_ann_info(self, idx):
        return self._parse_ann_info(self.img_infos[idx], self.ann_infos[idx])

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline"""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1  # all is 1?
            else:
                raise NotImplementedError

    def _filter_imgs(self, min_size=32):
        """Filter images too small"""
        valid_inds = []
        ids_with_ann = []
        for _ in self.ann_infos:
            if len(_) > 0:
                ids_with_ann.append(int(_[0]['image_id']))
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and int(img_info['id']) not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def load_imgs(self, idx_file, img_dir, calib_dir):
        """
        img_dir: such as 'training/image_2/000557.png'
        """
        img_infos = []
        with open(idx_file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                self.sampler_num += 1
                info = {}
                idx = line.strip()
                filename = img_dir + '/' + idx + '.png'
                info['filename'] = f'training/image_2/{idx}.png'
                info['file_name'] = f'training/image_2/{idx}.png'
                info['id'] = int(idx)
                img = mmcv.imread(filename)
                info['height'] = img.shape[0]
                info['width'] = img.shape[1]
                calib_filename = calib_dir + '/' + idx + '.txt'
                tri2v, trv2c, rect, cam_intrinsic = self.get_calib(calib_filename)
                info['Tri2v'] = tri2v
                info['Trv2c'] = trv2c
                info['rect'] = rect
                info['cam_intrinsic'] = cam_intrinsic
                img_infos.append(copy.deepcopy(info))
        return img_infos

    def load_anns(self, idx_file, ann_dir, calib_dir):
        ann_info = []
        id = 0
        with open(idx_file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                idx = line.strip()
                ann_filename = ann_dir + '/' + idx + '.txt'
                calib_filename = calib_dir + '/' + idx + '.txt'
                p2 = self.get_p(calib_filename)
                info_list = []
                with open(ann_filename, 'r') as f_ann:
                    ann_lines = f_ann.readlines()
                    for ann_line in ann_lines:
                        info = {}
                        instances = ann_line.split()
                        label = instances[0]
                        truncated = float(instances[1])
                        occluded = int(instances[2])
                        alpha = float(instances[3])
                        box_2d = [float(instance) for instance in instances[4:8]]
                        box_3d = [float(instance) for instance in instances[8:]]
                        # Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc DontCare
                        if label not in ['Car',]:
                            continue
                        x1, y1, x2, y2 = box_2d[0], box_2d[1], box_2d[2], box_2d[3]
                        h, w, l, x, y, z, ry = box_3d[0], box_3d[1], box_3d[2], box_3d[3], box_3d[4], \
                                               box_3d[5], box_3d[6]
                        box = [x1, y1, x2, y2, alpha, x, y, z, h, w, l, ry]
                        box3d_pts_2d, box3d_pts_3d = self.compute_box_3d(box, p2)
                        # TODO: when depth < 0.1m, box3d projected will be None
                        # if box3d_pts_2d is None:
                        #     continue
                        if alpha > 0 and alpha <= np.pi/2:
                            side_points, front_points = [6, 5, 1, 2], [4, 5, 1, 0]
                        elif alpha > np.pi/2 and alpha <= np.pi:
                            side_points, front_points = [4, 7, 3, 0], [4, 5, 1, 0]
                        elif alpha > -1* np.pi/2 and alpha <= 0:
                            side_points, front_points = [5, 6, 2, 1], [6, 7, 3, 2]
                        elif alpha > -1*np.pi and alpha <= -1*np.pi/2:
                            side_points, front_points = [4, 7, 3, 0], [6, 7, 3, 2]
                        scenter3d = np.mean(box3d_pts_3d[side_points], axis=0).reshape(1, 3)
                        fcenter3d = np.mean(box3d_pts_3d[front_points], axis=0).reshape(1, 3)
                        center3d = np.mean(box3d_pts_3d, 0).reshape(1, 3)
                        scenter2d = self.project_to_image(scenter3d, p2)[0]
                        fcenter2d = self.project_to_image(fcenter3d, p2)[0]
                        center2d = self.project_to_image(center3d, p2)[0]
                        proj_xmin = np.min(box3d_pts_2d[:, 0])
                        proj_xmax = np.max(box3d_pts_2d[:, 0])
                        proj_ymin = np.min(box3d_pts_2d[:, 1])
                        proj_ymax = np.max(box3d_pts_2d[:, 1])
                        info['file_name'] = f'training/image_2/{idx}.png'
                        info['image_id'] = int(idx)
                        info['area'] = (proj_ymax-proj_ymin)*(proj_xmax*proj_xmin)
                        info['category_name'] = label
                        info['category_id'] = self.cls2id[label]
                        info['bbox'] = [proj_xmin, proj_ymin, proj_xmax-proj_xmin, proj_ymax-proj_ymin]
                        info['iscrowd'] = 0
                        info['bbox_cam3d'] = [x, y, z, l, h, w, ry]
                        info['bbox_scam3d'] = [scenter3d[0][0], scenter3d[0][1], scenter3d[0][2], l, h, w, ry]
                        info['bbox_fcam3d'] = [fcenter3d[0][0], fcenter3d[0][1], fcenter3d[0][2], l, h, w, ry]
                        info['velo_cam3d'] = -1
                        info['center2d'] = [center2d[0], center2d[1], z]
                        info['scenter2d'] = [scenter2d[0], scenter2d[1], scenter3d[0][2]]
                        info['fcenter2d'] = [fcenter2d[0], fcenter2d[1], fcenter3d[0][2]]
                        info['attribute_name'] = -1
                        info['attribute_id'] = -1
                        info['segmentation'] = []
                        info['id'] = id
                        id = id + 1
                        info_list.append(copy.deepcopy(info))
                ann_info.append(copy.deepcopy(info_list))
        return ann_info

    def e_notation_to_float(self, numstr: str):
        """
        :param numstr: Scientific notation
        :return: double num
        """
        num = numstr.upper()
        assert 'E' in num

        e = num.find('E')

        big = num[:e]

        big = float(big)
        tmp = num[e + 1:]
        tmp = int(tmp)
        num_tmp = big * pow(10, tmp)
        return num_tmp

    def get_calib(self, calib_path):
        """
        :param p_path: the matrix path
        :return: Internal parameter matrix
        """
        with open(calib_path, 'r') as calib_file:
            lines = calib_file.readlines()
            # cam_intrinsic
            P2 = lines[2]
            assert P2[0:2] == 'P2'
            P2 = P2.split(' ')
            P2 = P2[1:]
            for tmp in range(12):
                P2[tmp] = self.e_notation_to_float(P2[tmp])
            cam_intrinsic = [[P2[0], P2[1], P2[2], P2[3]],
                             [P2[4], P2[5], P2[6], P2[7]],
                             [P2[8], P2[9], P2[10], P2[11]],
                             [0.0, 0.0, 0.0, 1.0]]
            # rect
            Rect = lines[4]
            assert Rect[0:7] == 'R0_rect'
            Rect = Rect.split(' ')
            Rect = Rect[1:]
            for tmp in range(9):
                Rect[tmp] = self.e_notation_to_float(Rect[tmp])
            rect = [[Rect[0], Rect[1], Rect[2], 0.0],
                    [Rect[3], Rect[4], Rect[5], 0.0],
                    [Rect[6], Rect[7], Rect[8], 0.0],
                    [0.0, 0.0, 0.0, 1.0]]
            # Trv2c
            Trv2c = lines[5]
            assert Trv2c[0:14] == 'Tr_velo_to_cam'
            Trv2c = Trv2c.split(' ')
            Trv2c = Trv2c[1:]
            for tmp in range(12):
                Trv2c[tmp] = self.e_notation_to_float(Trv2c[tmp])
            trv2c = [[Trv2c[0], Trv2c[1], Trv2c[2], Trv2c[3]],
                     [Trv2c[4], Trv2c[5], Trv2c[6], Trv2c[7]],
                     [Trv2c[8], Trv2c[9], Trv2c[10], Trv2c[11]],
                     [0.0, 0.0, 0.0, 1.0]]
            # Tri2v
            Tri2v = lines[6]
            assert Tri2v[0:14] == 'Tr_imu_to_velo'
            Tri2v = Tri2v.split(' ')
            Tri2v = Tri2v[1:]
            for tmp in range(12):
                Tri2v[tmp] = self.e_notation_to_float(Tri2v[tmp])
            tri2v = [[Tri2v[0], Tri2v[1], Tri2v[2], Tri2v[3]],
                     [Tri2v[4], Tri2v[5], Tri2v[6], Tri2v[7]],
                     [Tri2v[8], Tri2v[9], Tri2v[10], Tri2v[11]],
                     [0.0, 0.0, 0.0, 1.0]]
        return tri2v, trv2c, rect, cam_intrinsic

    def get_p(self, p_path):
        """
        :param p_path: the matrix path
        :return: Internal parameter matrix
        """
        with open(p_path, 'r') as f:
            lines = f.readlines()
            P2_line = lines[2]
            assert P2_line[0:2] == 'P2'
            P2_line = P2_line.split(' ')
            P2_line = P2_line[1:]
            for tmp in range(12):
                P2_line[tmp] = float(self.e_notation_to_float(P2_line[tmp]))
            p = np.reshape(P2_line, [3, 4])
        return p

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def convertRot2Alpha(self, ry3d, z3d, x3d):

        alpha = ry3d - math.atan2(x3d, z3d)
        while alpha > math.pi: alpha -= math.pi * 2
        while alpha < (-math.pi): alpha += math.pi * 2

        return alpha

    def convertAlpha2Rot(self, alpha, z3d, x3d):

        ry3d = alpha + math.atan2(x3d, z3d)  # + 0.5 * math.pi
        while ry3d > math.pi: ry3d -= math.pi * 2
        while ry3d < (-math.pi): ry3d += math.pi * 2

        return ry3d

    def roty(self, t):
        ''' Rotation about the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    def project_to_image(self, pts_3d, P):
        ''' Project 3d points to image plane.

        Usage: pts_2d = projectToImage(pts_3d, P)
          input: pts_3d: nx3 matrix
                 P:      3x4 projection matrix
          output: pts_2d: nx2 matrix

          P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
          => normalize projected_pts_2d(2xn)

          <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
              => normalize projected_pts_2d(nx2)
        '''
        n = pts_3d.shape[0]
        pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
        # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
        pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def compute_box_3d(self, obj, P):
        ''' Takes an object and a projection matrix (P) and projects the 3d
            bounding box into the image plane.
            Returns:
                corners_2d: (8,2) array in left image coord.
                corners_3d: (8,3) array in in rect camera coord.
        '''
        # [x1, y1, x2, y2, alpha, x, y, z, h, w, l, ry]
        # compute rotational matrix around yaw axis
        R = self.roty(obj[11])
        # 3d bounding box dimensions
        l = obj[10];
        w = obj[9];
        h = obj[8];

        # 3d bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + obj[5];
        corners_3d[1, :] = corners_3d[1, :] + obj[6];
        corners_3d[2, :] = corners_3d[2, :] + obj[7];
        # print 'cornsers_3d: ', corners_3d
        # TODO: only draw 3d bounding box for objs in front of the camera
        # if np.any(corners_3d[2, :] < 0.1):
        #     corners_2d = None
        #     return corners_2d, np.transpose(corners_3d)

        # project the 3d bounding box into the image plane
        corners_2d = self.project_to_image(np.transpose(corners_3d), P);
        # print 'corners_2d: ', corners_2d
        return corners_2d, np.transpose(corners_3d)

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not isinstance(outputs[0], dict):
            result_files = self.bbox2result_kitti2d(outputs, self.CLASSES,
                                                    pklfile_prefix,
                                                    submission_prefix)
        elif 'pts_bbox' in outputs[0] or 'img_bbox' in outputs[0] or \
                'img_bbox2d' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = submission_prefix + name
                else:
                    submission_prefix_ = None
                if '2d' in name:
                    result_files_ = self.bbox2result_kitti2d(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                else:
                    result_files_ = self.bbox2result_kitti(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import kitti_eval
        gt_annos = [info['annos'] for info in self.anno_infos]

        if isinstance(result_files, dict):
            ap_dict = dict()
            for name, result_files_ in result_files.items():
                eval_types = ['bbox', 'bev', '3d']
                if '2d' in name:
                    eval_types = ['bbox']
                ap_result_str, ap_dict_ = kitti_eval(
                    gt_annos,
                    result_files_,
                    self.CLASSES,
                    eval_types=eval_types)
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)

        else:
            if metric == 'img_bbox2d':
                ap_result_str, ap_dict = kitti_eval(
                    gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
            else:
                ap_result_str, ap_dict = kitti_eval(gt_annos, result_files,
                                                    self.CLASSES)
            print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show:
            self.show(results, out_dir)
        return ap_dict

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the \
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.anno_infos)
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.anno_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]

            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-np.arctan2(box[0], box[2]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print('Result is saved to %s' % out)

        return det_annos

    def bbox2result_kitti2d(self,
                            net_outputs,
                            class_names,
                            pklfile_prefix=None,
                            submission_prefix=None):
        """Convert 2D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the \
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries have the kitti format
        """
        assert len(net_outputs) == len(self.anno_infos)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])
            sample_idx = self.anno_infos[i]['image']['image_idx']

            num_example = 0
            for label in range(len(bboxes_per_sample)):
                bbox = bboxes_per_sample[label]
                for i in range(bbox.shape[0]):
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-10)
                    anno['bbox'].append(bbox[i, :4])
                    # set dimensions (height, width, length) to zero
                    anno['dimensions'].append(
                        np.zeros(shape=[3], dtype=np.float32))
                    # set the 3D translation to (-1000, -1000, -1000)
                    anno['location'].append(
                        np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                    anno['rotation_y'].append(0.0)
                    anno['score'].append(bbox[i, 4])
                    num_example += 1

            if num_example == 0:
                annos.append(
                    dict(
                        name=np.array([]),
                        truncated=np.array([]),
                        occluded=np.array([]),
                        alpha=np.array([]),
                        bbox=np.zeros([0, 4]),
                        dimensions=np.zeros([0, 3]),
                        location=np.zeros([0, 3]),
                        rotation_y=np.array([]),
                        score=np.array([]),
                    ))
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print('Result is saved to %s' % out)

        if submission_prefix is not None:
            # save file in submission format
            mmcv.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = self.anno_infos[i]['image']['image_idx']
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.
                - boxes_3d (:obj:`CameraInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.
                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in \
                    camera coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        img_shape = info['image']['image_shape']
        P2 = box_preds.tensor.new_tensor(P2)

        box_preds_camera = box_preds
        box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR,
                                               np.linalg.inv(rect @ Trv2c))

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                          (box_2d_preds[:, 1] < image_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds
        valid_inds = valid_cam_inds

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds_lidar[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)