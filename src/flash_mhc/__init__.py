"""flash-mHC: standalone high-performance mHC-lite in Torch+Triton."""

from .block import MHCLiteBlock, build_permutation_matrices
from . import ops as _ops  # Register torch.library custom ops.

__all__ = ["MHCLiteBlock", "build_permutation_matrices"]
