# optimizer
# This schedule is mainly used on S3DIS dataset in segmentation task
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.2, weight_decay=0.0001, momentum=0.9))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0.002,
        by_epoch=True,
        T_max=150)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=1)
