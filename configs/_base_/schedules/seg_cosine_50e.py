# optimizer
# This schedule is mainly used on S3DIS dataset in segmentation task
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.001))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=True,
        T_max=50)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
