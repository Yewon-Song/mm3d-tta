# Copyright (c) OpenMMLab. All rights reserved.
# nuScenes lidar semantic segmentation (7 super-classes via learning_map).
#
# This config is designed to be compatible with YAML-style `learning_map`
# where ignored labels are represented as 19. The dataset implementation
# (NuScenesSegDataset from projects/TPVFormer) will convert negative targets
# to `ignore_index` for training.

dataset_type = 'NuScenesSegDataset'
data_root = 'data/nuscenes/'

# 7 super-classes (example used in multiple cross-dataset settings)
class_names = [
    'Vehicle',     # 0
    'Pedestrian',  # 1
    'Road',        # 2
    'Sidewalk',    # 3
    'Terrain',     # 4
    'Manmade',     # 5
    'Vegetation',  # 6
]

IGNORE_INDEX = 255

# YAML `learning_map` from raw nuScenes lidarseg labels (0..31) -> 7 classes.
# Use 19 to indicate ignored labels; the dataset will map 19 -> IGNORE_INDEX.
label_mapping = {
    0: 19,
    1: 19,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 19,
    10: 19,
    11: 19,
    12: 19,
    13: 19,
    14: 19,
    15: 19,
    16: 19,
    17: 0,
    18: 19,
    19: 19,
    20: 0,
    21: 19,
    22: 19,
    23: 19,
    24: 2,
    25: 19,
    26: 3,
    27: 4,
    28: 5,
    29: 19,
    30: 6,
    31: 19,
}

metainfo = dict(
    classes=class_names,
    seg_label_mapping=label_mapping,
    ignore_index=IGNORE_INDEX,
)

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='lidarseg/v1.0-trainval',
)

input_modality = dict(use_lidar=True, use_camera=False)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        dataset_type='nuscenes',
    ),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        dataset_type='nuscenes',
    ),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points']),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        data_prefix=data_prefix,
        modality=input_modality,
        filter_empty_gt=False,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        data_prefix=data_prefix,
        modality=input_modality,
        test_mode=True,
    ),
)

val_dataloader = test_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator


