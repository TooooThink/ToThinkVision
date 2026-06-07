"""Monkey-patch torch.Tensor.__array__ for numpy 1.26+ compatibility.

Problem
-------
numpy >=1.26 calls ``obj.__array__(dtype=…, copy=False)``.  Older PyTorch
versions (< 2.1) define ``__array__(self, dtype=None)`` without the ``copy``
keyword, which raises a ``TypeError``.  numpy catches the ``TypeError`` and
retries without ``copy``, but the failed call can leave the C-level array
conversion in a bad state — particularly for CUDA tensors — causing a
SIGSEGV inside the torch/numpy C extension.

Fix
---
Wrap ``torch.Tensor.__array__`` to silently accept and discard the ``copy``
keyword.  This lets numpy's fast path succeed without the retry, avoiding
the crash.

This file is loaded automatically by Python when its parent directory is
in ``PYTHONPATH`` (via the standard ``sitecustomize`` mechanism).
"""
import inspect
import sys

try:
    import torch
except ImportError:
    sys.exit(0)  # torch not available — nothing to patch

_orig_array = torch.Tensor.__array__
_sig = inspect.signature(_orig_array)

# Only patch if the original __array__ doesn't already accept 'copy'
if "copy" not in _sig.parameters:

    def _patched_array(self, dtype=None, copy=None):
        # For CUDA tensors, torch's __array__ calls .cpu().numpy() internally.
        # We just need to avoid passing copy= to the original.
        if dtype is not None:
            return _orig_array(self, dtype=dtype)
        return _orig_array(self)

    torch.Tensor.__array__ = _patched_array
