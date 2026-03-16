"""
Compatibility layer for importing mamba_ssm without CUDA extensions.
Patches mamba_ssm imports to allow loading on CPU-only environments.
The actual CUDA ops are required at training/inference time on GPU.

IMPORTANT: Environment requirements for the Mamba encoder:
  - PyTorch >= 1.13.1 (with CUDA 11.7+)
  - transformers == 4.36.2 (NOT 5.x — mamba_ssm uses removed APIs)
  - triton == 2.2.0
  - causal-conv1d == 1.1.1
  See requirements_mamba.txt for the full list.
"""
import sys
import types
import warnings


def _check_versions():
    """Warn about known incompatible dependency versions."""
    try:
        import transformers
        major = int(transformers.__version__.split('.')[0])
        if major >= 5:
            warnings.warn(
                f"transformers {transformers.__version__} detected, but mamba_ssm "
                f"requires transformers<5.0 (e.g., 4.36.2). "
                f"Run: pip install transformers==4.36.2",
                UserWarning,
                stacklevel=3,
            )
    except ImportError:
        pass


def ensure_mamba_importable(mamba_path=None):
    """Make mamba_ssm importable by adding path and creating CUDA stubs if needed."""
    _check_versions()

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
