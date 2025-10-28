"""
fp_quant: A Python package for fp_quant
"""

from .module import FPQuantLinear
from .utils import (
    FPQuantConfig,
    FPQuantDtype,
    replace_with_fp_quant_linear,
    replace_quantize_with_fp_quant_linear,
    finalize_master_weights,
)
