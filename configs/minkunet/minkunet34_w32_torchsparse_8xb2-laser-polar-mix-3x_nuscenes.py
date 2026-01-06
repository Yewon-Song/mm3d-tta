_base_ = [
    '../_base_/datasets/nus_seg.py',
    '../_base_/models/minkunet.py',
    '../_base_/schedules/schedule-3x.py',
    '../_base_/default_runtime.py'
]

# =========================================================
# nuScenes lidarseg -> 7-class mapping (YAML learning_map 기준)
# =========================================================
IGNORE_INDEX = 255

label_mapping = {
    0: IGNORE_INDEX,   # noise
    1: IGNORE_INDEX,   # animal

    2: 1,  # human.pedestrian.adult -> pedestrian
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,

    9: IGNORE_INDEX,   # barrier
    10: IGNORE_INDEX,
    11: IGNORE_INDEX,
    12: IGNORE_INDEX,
    13: IGNORE_INDEX,

    14: IGNORE_INDEX,  # bicycle
    15: IGNORE_INDEX,  # bus
    16: IGNORE_INDEX,

    17: 0,             # vehicle.car -> car
    18: IGNORE_INDEX,  # construction

    19: IGNORE_INDEX,
    20: 0,             # police -> car (YAML 기준)
    21: IGNORE_INDEX,
    22: IGNORE_INDEX,
    23: IGNORE_INDEX,

    24: 2,             # driveable_surface
    25: IGNORE_INDEX,
    26: 3,             # sidewalk
    27: 4,             # terrain
    28: 5,             # manmade
    29: IGNORE_INDEX,
    30: 6,             # vegetation
    31: IGNORE_INDEX,  # ego
}

# =========================================================
# 7-class definition
# =========================================================
class_names = [
    'noise',
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'movable_object.trafficcone',
    'static_object.bicycle_rack',
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck',
    'flat.driveable_surface',
    'flat.other',
    'flat.sidewalk',
    'flat.terrain',
    'static.manmade',
    'static.other',
    'static.vegetation',
    'vehicle.ego'
]

metainfo = dict(
    classes=class_names,
    seg_label_mapping=label_mapping,
    ignore_index=IGNORE_INDEX
)

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='lidarseg/v1.0-trainval'
)

model = dict(
    data_preprocessor=dict(
        max_voxels=160000,   # 네가 이미 쓰던 값 유지 가능
        voxel_layer=dict(
            voxel_size=[0.1, 0.1, 0.1],
            point_cloud_range=[-50, -50, -5, 50, 50, 3]
        )
    ),
    backbone=dict(
        encoder_blocks=[2, 3, 4, 6]
    ),
    decode_head = dict(
    type='MinkUNetHead',
    num_classes=32,
    loss_decode=dict(
        type='mmdet.CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0,
        avg_non_ignore=True
    ),
    ignore_index=255
    )
)


# =========================================================
# Pipelines
# =========================================================
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int8',   # ⭐ 핵심
        dataset_type='nuscenes'
    ),
    dict(type='PointSegClassMapping'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]




val_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int64',   
        dataset_type='nuscenes'
    ),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_dataloader = dict(
    dataset=dict(
        type='NuScenesSegDataset',  
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        data_prefix=data_prefix,
        filter_empty_gt=False
    )
)

val_dataloader = dict(
    dataset=dict(
        type='NuScenesSegDataset',  
        ann_file='nuscenes_infos_val.pkl',
        pipeline=val_pipeline,
        metainfo=metainfo,
        data_prefix=data_prefix,
        test_mode=True
    )
)

test_dataloader = val_dataloader

# =========================================================
# Evaluator
# =========================================================
val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))
