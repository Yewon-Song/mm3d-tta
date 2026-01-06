# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Union

from mmengine.dataset import BaseDataset

from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class NuScenesSegDataset(BaseDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        test_mode (bool): Store `True` when building test or val dataset.
    """
    METAINFO = {
        'classes':
        ('noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'),
        'ignore_index':
        0,
        'label_mapping':
        dict([(1, 0), (5, 0), (7, 0), (8, 0), (10, 0), (11, 0), (13, 0),
              (19, 0), (20, 0), (0, 0), (29, 0), (31, 0), (9, 1), (14, 2),
              (15, 3), (16, 3), (17, 4), (18, 5), (21, 6), (2, 7), (3, 7),
              (4, 7), (6, 7), (12, 8), (22, 9), (23, 10), (24, 11), (25, 12),
              (26, 13), (27, 14), (28, 15), (30, 16)]),
        'palette': [
            [0, 0, 0],  # noise
            [255, 120, 50],  # barrier              orange
            [255, 192, 203],  # bicycle              pink
            [255, 255, 0],  # bus                  yellow
            [0, 150, 245],  # car                  blue
            [0, 255, 255],  # construction_vehicle cyan
            [255, 127, 0],  # motorcycle           dark orange
            [255, 0, 0],  # pedestrian           red
            [255, 240, 150],  # traffic_cone         light yellow
            [135, 60, 0],  # trailer              brown
            [160, 32, 240],  # truck                purple
            [255, 0, 255],  # driveable_surface    dark pink
            [139, 137, 137],  # other_flat           dark red
            [75, 0, 75],  # sidewalk             dard purple
            [150, 240, 80],  # terrain              light green
            [230, 230, 250],  # manmade              white
            [0, 175, 0],  # vegetation           green
        ]
    }

    def _normalize_seg_label_mapping(self, label_mapping: dict,
                                     ignore_index: int) -> dict:
        """Normalize segmentation label mapping.

        This dataset is often used with a custom `learning_map` where
        ignored labels are represented as -1 (as in many lidarseg YAML files).
        However, loss functions in MMDetection3D typically expect ignored
        labels to be `ignore_index` (e.g., 255).

        Args:
            label_mapping (dict): Mapping from raw label id -> target label id.
                The target label id may be -1 to indicate ignored labels.
            ignore_index (int): The label id used as ignore index in training.

        Returns:
            dict: A cleaned mapping with int keys/values where any negative
                target label is replaced by `ignore_index`.
        """
        normalized = {}
        for k, v in label_mapping.items():
            kk = int(k)
            vv = int(v)
            if vv < 0:
                vv = int(ignore_index)
            normalized[kk] = vv
        return normalized

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 modality: dict = None,
                 filter_empty_gt: bool = True,
                 **kwargs) -> None:
        # NOTE:
        # Historically, this dataset ignored config-provided `metainfo` and
        # always used the hard-coded 17-class mapping above.
        # For cross-dataset experiments and super-class training (e.g., 7-class
        # `learning_map`), we must respect `metainfo` from config.
        metainfo_cfg = kwargs.pop('metainfo', None)
        kwargs.pop('modality', None)
        kwargs.pop('filter_empty_gt', None)

        metainfo = dict(metainfo_cfg) if metainfo_cfg is not None else dict()
        classes = metainfo.get('classes', None)
        if classes is None:
            classes = self.METAINFO['classes']
            metainfo['classes'] = classes

        # Prefer config-provided mapping. Support both names:
        # - seg_label_mapping: used by PointSegClassMapping
        # - label_mapping: legacy field used by this dataset
        mapping = metainfo.get('seg_label_mapping', None)
        if mapping is None:
            mapping = metainfo.get('label_mapping', None)
        if mapping is None:
            mapping = self.METAINFO.get('label_mapping', {})

        ignore_index = int(metainfo.get('ignore_index',
                                       self.METAINFO.get('ignore_index', 255)))
        metainfo['ignore_index'] = ignore_index

        if isinstance(mapping, dict):
            mapping = self._normalize_seg_label_mapping(mapping, ignore_index)
        metainfo['seg_label_mapping'] = mapping
        metainfo['label_mapping'] = mapping

        # Used by SegMetric to print per-class results.
        metainfo['label2cat'] = {i: cat_name for i, cat_name in enumerate(classes)}

        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """

        data_list = []
        info['lidar_points']['lidar_path'] = \
            osp.join(
                self.data_prefix.get('pts', ''),
                info['lidar_points']['lidar_path'])

        for cam_id, img_info in info['images'].items():
            if 'img_path' in img_info:
                if cam_id in self.data_prefix:
                    cam_prefix = self.data_prefix[cam_id]
                else:
                    cam_prefix = self.data_prefix.get('img', '')
                img_info['img_path'] = osp.join(cam_prefix,
                                                img_info['img_path'])

        if 'pts_semantic_mask_path' in info:
            info['pts_semantic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_semantic_mask', ''),
                         info['pts_semantic_mask_path'])

        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        info['seg_label_mapping'] = self.metainfo.get(
            'seg_label_mapping', self.metainfo.get('label_mapping', {}))
        # Provide ignore_index to PointSegClassMapping so that ignored labels
        # become the expected ignore id for loss functions.
        info['ignore_index'] = int(self.metainfo.get('ignore_index', 255))

        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode:
            info['eval_ann_info'] = dict()

        data_list.append(info)
        return data_list
