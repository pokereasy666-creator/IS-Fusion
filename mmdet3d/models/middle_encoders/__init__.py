# Copyright (c) OpenMMLab. All rights reserved.
from .pillar_scatter import PointPillarsScatter
from .sparse_encoder import SparseEncoder

from .sparse_unet import SparseUNet

from .fusion_encoder import ISFusionEncoder
from ..sst.sst_input_layer_v2 import SSTInputLayerV2
try:
    from .mamba_encoder import MambaMiddleEncoder
except ImportError as exc:
    from ..builder import MIDDLE_ENCODERS

    _MAMBA_ENCODER_IMPORT_ERROR = exc

    @MIDDLE_ENCODERS.register_module()
    class MambaMiddleEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                'MambaMiddleEncoder could not be imported. Install the '
                'Mamba dependencies from requirements_mamba.txt and ensure '
                'the PointMamba utilities can be imported.'
            ) from _MAMBA_ENCODER_IMPORT_ERROR

__all__ = ['PointPillarsScatter', 'SparseEncoder', 'SparseUNet', 'ISFusionEncoder', 'SSTInputLayerV2', 'MambaMiddleEncoder']
