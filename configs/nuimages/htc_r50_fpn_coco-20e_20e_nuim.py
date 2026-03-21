_base_ = './htc_r50_fpn_coco-20e_1x_nuim.py'
# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[16, 19], gamma=0.1, by_epoch=True)]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
