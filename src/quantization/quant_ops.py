from typing import Tuple, Callable

import torch

from .quant_args import QuantizationFormat, ScalePrecision

### Constants
FP4_E2M1_MAX = 6
FP8_E4M3_MAX = 448
NVFP_GROUPSIZE = 16
MXFP_GROUPSIZE = 32
FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

FP4_GRID =  [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_BITPACKING_PERM = [15, 14, 13, 12, 11, 10,  9,  8,  0,  1,  2,  3,  4,  5,  6,  7]
FP4_SCALE = 3 / 4

### Common utils ###
def get_quantization_range(format: QuantizationFormat, bits: int, symmetric: bool) -> Tuple[int, int]:
    if format in [QuantizationFormat.FP, QuantizationFormat.NVFP, QuantizationFormat.MXFP]:
        assert bits == 4, "Currently only 4-bit NVFP is supported"
        return -FP4_E2M1_MAX, FP4_E2M1_MAX
    elif format == QuantizationFormat.INT:
        bit_range = 2 ** bits
        if symmetric:
            q_min, q_max = -bit_range // 2, bit_range // 2 - 1
        else:
            q_min, q_max = 0, bit_range - 1
        return q_min, q_max
def pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(FP4_GRID, device=x.device)
    perm = torch.tensor(FP4_BITPACKING_PERM, device=x.device)

    grid_ids = torch.bucketize(x, grid)
    lo, hi = (grid_ids - 1).clamp(min=0, max=2 ** 4 - 1), grid_ids.clamp(min=0, max=2 ** 4 - 1)
    g_lo, g_hi = grid[lo], grid[hi]
    pick_hi = (g_hi - x < x - g_lo) | (g_hi - x == x - g_lo) & (perm[hi] % 2 == 0)
    q = torch.where(pick_hi, perm[hi], perm[lo])
    return (q[:, 1::2] << 4 | q[:, ::2]).to(torch.uint8)
### Integer Quantization ###
def quantize_int(x: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, q_min: int, q_max: int) -> torch.Tensor:
    return (x / scales + zeros).round().clamp(q_min, q_max)

def dequantize_int(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    return q.sub(zeros).mul(scales)

def quantize_dequantize_int(x: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, q_min: int, q_max: int):
    xq = dequantize_int(quantize_int(x, scales, zeros, q_min, q_max), scales, zeros)
    return x + (xq - x).detach()

### Float Quantization ###

# TODO

### NVFP Quantization ###
def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign

def quantize_fp4(x: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, q_min: int, q_max: int):
    return cast_to_fp4(x / scales)

def dequantize_fp4(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
    return q.mul(scales)

def quantize_dequantize_fp4(x: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, q_min: int, q_max: int):
    xq = dequantize_fp4(quantize_fp4(x, scales, zeros, q_min, q_max), scales, zeros)
    return x + (xq - x).detach()


#### MXFP quantization
def cast_to_eBm0(x: torch.Tensor, ebits: int, emax: int):
    """
    Args:
        x: input tensor
        ebits: number of exponent bits
        emax: maximum exponent value for element data format
    """
    assert ebits % 2 == 0, "EBm0 expects even number of bits"
    assert x.ge(0).all(), "EBm0 expects positive inputs"
    qmin = -(2 ** (ebits - 1) - 1)
    qmax = +(2 ** (ebits - 1) - 1)
    # We clamp values instead of overflow (see https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/mx_ops.py#L83)
    return 2 ** (x.clamp(min=FP32_MIN_NORMAL).log2().floor().clamp(qmin, qmax) - emax)


### Quantization functions factory ###
def get_quantization_fns(format: QuantizationFormat, bits: int) -> Tuple[Callable, Callable, Callable]:
    if format in  [QuantizationFormat.FP, QuantizationFormat.NVFP, QuantizationFormat.MXFP]:
        if bits == 4:
            return quantize_fp4, dequantize_fp4, quantize_dequantize_fp4
    if format == QuantizationFormat.INT:
        return quantize_int, dequantize_int, quantize_dequantize_int
    raise ValueError(
        f"Unsupported quantization configuration\n"
        f"format: {format}\n"
        f"bits: {bits}\n"
    )

### Saving utilities ###
def cast_scales_to_eXmY(scales: torch.Tensor, scale_precision: str) -> torch.Tensor:
    scale_precision = ScalePrecision(scale_precision)
    if scale_precision == ScalePrecision.E4M3:
        return scales.to(torch.float8_e4m3fn).view(torch.uint8)
    elif scale_precision == ScalePrecision.E8M0:
        # 2 is EMAX
        scales = scales.to(torch.float8_e8m0fnu).view(torch.uint8)
    return scales
