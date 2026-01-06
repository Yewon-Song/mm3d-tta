# Copyright (c) OpenMMLab. All rights reserved.
# nuScenes lidar segmentation base config (for configs to _base_/datasets)

# nuScenes semantic segmentation (lidarseg) common setting
dataset_type = 'NuScenesSegDataset'
data_root = 'data/nuscenes/'

# metainfo used by datasets and models
metainfo = dict(classes=class_names, seg_label_mapping=label_mapping, ignore_index=255)

# data prefix: where point clouds and semantic masks live relative to data_root
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='lidarseg/v1.0-trainval')

input_modality = dict(use_lidar=True, use_camera=False)

# Pipelines (use string 'type' names so config merging doesn't require imports)
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        dataset_type='nuscenes'),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        dataset_type='nuscenes'),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# construct a pipeline for data and gt loading in show function
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Test-time augmentation (optional)
tta_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        dataset_type='nuscenes'),
    dict(type='PointSegClassMapping'),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0,
                flip_ratio_bev_vertical=1.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
                    [dict(type='Pack3DDetInputs', keys=['points'])]])
]

# Dataloaders
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NuScenesSegDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        data_prefix=data_prefix,
        modality=input_modality,
        filter_empty_gt=False))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesSegDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        data_prefix=data_prefix,
        modality=input_modality,
        test_mode=True))

val_dataloader = test_dataloader

# Evaluators
val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator
