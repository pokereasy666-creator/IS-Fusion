# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
lr = 0.008  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        milestones=[24, 32],
        gamma=0.1,
        by_epoch=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
