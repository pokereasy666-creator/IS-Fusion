# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet3d.registry import (
    MODELS,
    BACKBONES,
    DETECTORS,
    HEADS,
    LOSSES,
    NECKS,
    ROI_EXTRACTORS,
    SHARED_HEADS,
    SEGMENTORS,
    VOXEL_ENCODERS,
    MIDDLE_ENCODERS,
    FUSION_LAYERS,
    VTRANSFORMS,
)


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_vtransform(cfg):
    return VTRANSFORMS.build(cfg)


def build_roi_extractor(cfg):
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field'
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field'
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field'
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field'
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_model(cfg, train_cfg=None, test_cfg=None):
    if cfg.type in ['EncoderDecoder3D']:
        return build_segmentor(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    else:
        return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)


def build_voxel_encoder(cfg):
    return VOXEL_ENCODERS.build(cfg)


def build_middle_encoder(cfg):
    return MIDDLE_ENCODERS.build(cfg)


def build_fusion_layer(cfg):
    return FUSION_LAYERS.build(cfg)
