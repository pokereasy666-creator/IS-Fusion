"""
MixerModel and Block adapted from PointMamba.
"""
import os
import math
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import Mamba directly from the module file, bypassing mamba_ssm.__init__
# which has fragile dependencies (transformers, CUDA extensions).
from .mamba_compat import ensure_mamba_importable
_mamba_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'PointMamba', 'mamba')
_mamba_path = os.path.normpath(_mamba_path)
if os.path.isdir(_mamba_path):
    ensure_mamba_importable(_mamba_path)
else:
    ensure_mamba_importable()

# Import Mamba class directly from its module to avoid __init__.py import chain
import importlib.util as _ilu
def _import_from_file(mod_name, file_path):
    """Import a module from file path, bypassing package __init__."""
    import sys
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = _ilu.spec_from_file_location(mod_name, file_path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Ensure selective_scan_interface is loaded first (Mamba depends on it)
_ssm_ops_path = os.path.join(_mamba_path, 'mamba_ssm', 'ops', 'selective_scan_interface.py')
if os.path.isfile(_ssm_ops_path):
    _import_from_file('mamba_ssm.ops.selective_scan_interface', _ssm_ops_path)

_mamba_simple_path = os.path.join(_mamba_path, 'mamba_ssm', 'modules', 'mamba_simple.py')
if os.path.isfile(_mamba_simple_path):
    _mamba_mod = _import_from_file('mamba_ssm.modules.mamba_simple', _mamba_simple_path)
    Mamba = _mamba_mod.Mamba
else:
    from mamba_ssm.modules.mamba_simple import Mamba

try:
    _ln_path = os.path.join(_mamba_path, 'mamba_ssm', 'ops', 'triton', 'layernorm.py')
    if os.path.isfile(_ln_path):
        _ln_mod = _import_from_file('mamba_ssm.ops.triton.layernorm', _ln_path)
        RMSNorm = _ln_mod.RMSNorm
        layer_norm_fn = _ln_mod.layer_norm_fn
        rms_norm_fn = _ln_mod.rms_norm_fn
    else:
        from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except (ImportError, Exception):
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from timm.models.layers import DropPath


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
            residual_in_fp32=False, drop_path=0.
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None,
            inference_params=None
    ):
        hidden_states = hidden_states + self.drop_path(
            self.mixer(self.norm(hidden_states), inference_params=inference_params))
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out: float = 0.,
            drop_path=0.,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids + pos

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, inference_params=inference_params
            )
            hidden_states = self.drop_out(hidden_states)

        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))
        return hidden_states
