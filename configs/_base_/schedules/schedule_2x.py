# optimizer
# This schedule is mainly used by models on nuScenes dataset
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    # max_norm=10 is better for SECOND
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        milestones=[20, 23],
        gamma=0.1,
        by_epoch=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
