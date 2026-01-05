# Copyright (c) OpenMMLab. All rights reserved.
from .dataset_wrappers import CBGSDataset
from .det3d_dataset import Det3DDataset
from .kitti_dataset import KittiDataset
from .lyft_dataset import LyftDataset
from .nuscenes_dataset import NuScenesDataset
try:
    # Import segmentation dataset adapter from projects if available to
    # register NuScenesSegDataset in DATASETS registry. This import is
    # optional and will be skipped silently if the projects package is
    # not present in PYTHONPATH.
    from .nuScenes_seg_adapter import NuScenesSegDataset  # noqa: F401
except Exception:
    # projects/TPVFormer may not be present; NuScenesSegDataset will not
    # be registered in that case. If you need segmentation dataset, make
    # sure the project is importable or add an implementation here.
    pass
# yapf: enable
from .s3dis_dataset import S3DISDataset, S3DISSegDataset
from .scannet_dataset import (ScanNetDataset, ScanNetInstanceSegDataset,
                              ScanNetSegDataset)
from .seg3d_dataset import Seg3DDataset
from .semantickitti_dataset import SemanticKittiDataset
from .sunrgbd_dataset import SUNRGBDDataset
# yapf: disable
from .transforms import (AffineResize, BackgroundPointsFilter, GlobalAlignment,
                         GlobalRotScaleTrans, IndoorPatchPointSample,
                         IndoorPointSample, LoadAnnotations3D,
                         LoadPointsFromDict, LoadPointsFromFile,
                         LoadPointsFromMultiSweeps, NormalizePointsColor,
                         ObjectNameFilter, ObjectNoise, ObjectRangeFilter,
                         ObjectSample, PointSample, PointShuffle,
                         PointsRangeFilter, RandomDropPointsColor,
                         RandomFlip3D, RandomJitterPoints, RandomResize3D,
                         RandomShiftScale, Resize3D, VoxelBasedPointSampler)
from .utils import get_loading_pipeline
from .waymo_dataset import WaymoDataset

__all__ = [
    'KittiDataset', 'CBGSDataset', 'NuScenesDataset', 'LyftDataset',
    'ObjectSample', 'RandomFlip3D', 'ObjectNoise', 'GlobalRotScaleTrans',
    'PointShuffle', 'ObjectRangeFilter', 'PointsRangeFilter',
    'LoadPointsFromFile', 'S3DISSegDataset', 'S3DISDataset',
    'NormalizePointsColor', 'IndoorPatchPointSample', 'IndoorPointSample',
    'PointSample', 'LoadAnnotations3D', 'GlobalAlignment', 'SUNRGBDDataset',
    'ScanNetDataset', 'ScanNetSegDataset', 'ScanNetInstanceSegDataset',
    'SemanticKittiDataset', 'Det3DDataset', 'Seg3DDataset',
    'LoadPointsFromMultiSweeps', 'WaymoDataset', 'BackgroundPointsFilter',
    'VoxelBasedPointSampler', 'get_loading_pipeline', 'RandomDropPointsColor',
    'RandomJitterPoints', 'ObjectNameFilter', 'AffineResize',
    'RandomShiftScale', 'LoadPointsFromDict', 'Resize3D', 'RandomResize3D',
]
