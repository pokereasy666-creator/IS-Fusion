# Copyright (c) OpenMMLab. All rights reserved.
try:
    from mmdet.core.bbox.samplers import (BaseSampler, CombinedSampler,
                                          InstanceBalancedPosSampler,
                                          IoUBalancedNegSampler, OHEMSampler,
                                          PseudoSampler, RandomSampler,
                                          SamplingResult)
except ImportError:
    from mmdet.models.task_modules.samplers import (BaseSampler, CombinedSampler,
                                                    PseudoSampler, RandomSampler,
                                                    SamplingResult)
    try:
        from mmdet.models.task_modules.samplers import (InstanceBalancedPosSampler,
                                                        IoUBalancedNegSampler,
                                                        OHEMSampler)
    except ImportError:
        InstanceBalancedPosSampler = None
        IoUBalancedNegSampler = None
        OHEMSampler = None

from .iou_neg_piecewise_sampler import IoUNegPiecewiseSampler

__all__ = [
    'BaseSampler', 'PseudoSampler', 'RandomSampler',
    'InstanceBalancedPosSampler', 'IoUBalancedNegSampler', 'CombinedSampler',
    'OHEMSampler', 'SamplingResult', 'IoUNegPiecewiseSampler'
]
