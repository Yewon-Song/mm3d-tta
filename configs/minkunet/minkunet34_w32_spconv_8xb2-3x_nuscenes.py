_base_ = [
    './minkunet34_w32_torchsparse_8xb2-laser-polar-mix-3x_nuscenes.py'
]

model = dict(
    data_preprocessor=dict(batch_first=True),
    backbone=dict(sparseconv_backend='spconv'))
    
optim_wrapper = dict(type='OptimWrapper')