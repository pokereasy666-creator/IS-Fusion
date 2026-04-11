# Copyright (c) OpenMMLab. All rights reserved.
from .overwrite_spconv.write_spconv2 import register_spconv2

try:
    import spconv
except ImportError:
    IS_SPCONV2_AVAILABLE = False
else:
    if hasattr(spconv, '__version__') and spconv.__version__ >= '2.0.0':
        IS_SPCONV2_AVAILABLE = register_spconv2()
    else:
        IS_SPCONV2_AVAILABLE = False

# Re-export commonly used spconv classes so that code doing
# ``from mmdet3d.ops import spconv; spconv.SparseSequential(...)``
# works transparently.
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import (SparseConvTensor, SparseSequential,
                                SparseModule, SparseMaxPool3d,
                                SparseConv2d, SparseConv3d,
                                SubMConv2d, SubMConv3d)

__all__ = ['IS_SPCONV2_AVAILABLE']
