"""
Compatibility layer for importing mamba_ssm without CUDA extensions.
Patches mamba_ssm imports to allow loading on CPU-only environments.
The actual CUDA ops are required at training/inference time on GPU.
"""
import sys
import types


def ensure_mamba_importable(mamba_path=None):
    """Make mamba_ssm importable by adding path and creating CUDA stubs if needed."""
    if mamba_path:
        if mamba_path not in sys.path:
            sys.path.insert(0, mamba_path)

    # Create stub for selective_scan_cuda if not available
    if 'selective_scan_cuda' not in sys.modules:
        try:
            import selective_scan_cuda  # noqa: F401
        except ImportError:
            stub = types.ModuleType('selective_scan_cuda')
            stub.__doc__ = "Stub module — build CUDA extensions for actual use."
            sys.modules['selective_scan_cuda'] = stub

    # Create stub for causal_conv1d_cuda if not available
    if 'causal_conv1d_cuda' not in sys.modules:
        try:
            import causal_conv1d_cuda  # noqa: F401
        except ImportError:
            stub = types.ModuleType('causal_conv1d_cuda')
            sys.modules['causal_conv1d_cuda'] = stub
