# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np

from mmdet3d.datasets import SemanticKittiDataset
from mmdet3d.utils import register_all_modules


def _generate_semantickitti_dataset_config():
    data_root = './tests/data/semantickitti/'
    ann_file = 'semantickitti_infos.pkl'
    classes = ('car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
               'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
               'other-ground', 'building', 'fence', 'vegetation', 'trunck',
               'terrian', 'pole', 'traffic-sign')

    seg_label_mapping = {
        0:  -1,  # "unlabeled"
        1:  -1,  # "outlier" -> unlabeled
        10:  0,  # "car"
        11: -1,  # "bicycle"
        13: -1,  # "bus" -> other-vehicle
        15: -1,  # "motorcycle"
        16: -1,  # "on-rails" -> other-vehicle
        18: -1,  # "truck"
        20: -1,  # "other-vehicle"
        30:  1,  # "person"
        31: -1,  # "bicyclist"
        32: -1,  # "motorcyclist"
        40:  2,  # "road"
        44:  2,  # "parking"
        48:  3,  # "sidewalk"
        49: -1,  # "other-ground"
        50:  5,  # "building"
        51:  5,  # "fence"
        52: -1,  # "other-structure" -> unlabeled
        60:  2,  # "lane-marking" -> road
        70:  6,  # "vegetation"
        71:  6,  # "trunk"
        72:  4,  # "terrain"
        80:  5,  # "pole"
        81:  5,  # "traffic-sign"
        99: -1,  # "other-object" -> unlabeled
        252: 0,  # "moving-car" -> car
        253: -1, # "moving-bicyclist"
        254: 1,  # "moving-person" -> person
        255: -1, # "moving-motorcyclist"
        256: -1, # "moving-on-rails" -> other-vehicle
        257: -1, # "moving-bus" -> other-vehicle
        258: -1, # "moving-truck"
        259: -1  # "moving-other-vehicle"
    }
    max_label = 259
    modality = dict(use_lidar=True, use_camera=False)
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            shift_height=True,
            load_dim=4,
            use_dim=[0, 1, 2]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True,
            seg_3d_dtype='np.int32'),
        dict(type='PointSegClassMapping'),
        dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
    ]

    data_prefix = dict(
        pts='sequences/00/velodyne', pts_semantic_mask='sequences/00/labels')

    return (data_root, ann_file, classes, data_prefix, pipeline, modality,
            seg_label_mapping, max_label)


class TestSemanticKittiDataset(unittest.TestCase):

    def test_semantickitti(self):
        (data_root, ann_file, classes, data_prefix, pipeline, modality,
         seg_label_mapping,
         max_label) = _generate_semantickitti_dataset_config()

        register_all_modules()
        np.random.seed(0)
        semantickitti_dataset = SemanticKittiDataset(
            data_root,
            ann_file,
            metainfo=dict(
                classes=classes,
                seg_label_mapping=seg_label_mapping,
                max_label=max_label),
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality)

        input_dict = semantickitti_dataset.prepare_data(0)

        points = input_dict['inputs']['points']
        data_sample = input_dict['data_samples']
        pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
        self.assertEqual(points.shape[0], pts_semantic_mask.shape[0])

        expected_pts_semantic_mask = np.array([
            12, 12, 12, 14, 14, 12, 19, 12, 14, 12, 12, 14, 15, 19, 14, 12, 12,
            12, 12, 19, 12, 12, 12, 12, 12, 14, 12, 15, 12, 14, 14, 17, 12, 14,
            14, 14, 15, 14, 12, 12, 14, 12, 17, 14, 12, 14, 12, 14, 14, 12
        ])

        self.assertTrue(
            (pts_semantic_mask.numpy() == expected_pts_semantic_mask).all())
