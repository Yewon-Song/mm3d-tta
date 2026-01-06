_base_ = [
    './minkunet34_w32_torchsparse_8xb2-laser-polar-mix-3x_semantickitti.py'
]

model = dict(
    data_preprocessor=dict(batch_first=True),
    backbone=dict(sparseconv_backend='spconv'))

optim_wrapper = dict(type='OptimWrapper')

# Evaluation-only: merge SemanticKITTI 19 classes into 7 super-classes for logging.
# This does NOT change training targets or model output channels.
eval_class_names = [
    'Vehicle',     # 0
    'Pedestrian',  # 1
    'Road',        # 2
    'Sidewalk',    # 3
    'Terrain',     # 4
    'Manmade',     # 5
    'Vegetation',  # 6
]

# IMPORTANT:
# In this experiment, GT/PRED labels follow the output of `labels_map` defined in
# `configs/_base_/datasets/semantickitti.py` (after PointSegClassMapping), i.e.:
# 0 car, 1 person, 2 road/parking/lane-marking, 3 sidewalk, 4 terrain,
# 5 manmade (building/fence/pole/traffic-sign), 6 vegetation/trunk.
eval_label_mapping = {
    0: 0,  # Vehicle
    1: 1,  # Pedestrian
    2: 2,  # Road
    3: 3,  # Sidewalk
    4: 4,  # Terrain
    5: 5,  # Manmade
    6: 6,  # Vegetation
}

val_evaluator = dict(type='SegMetric', class_names=eval_class_names, eval_label_mapping=eval_label_mapping)
test_evaluator = val_evaluator