from typing import Tuple, Optional

import torch

from .quant_args import QuantizationFormat, QuantizationGranularity, QuantizationObserver, ScalePrecision
from .quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX, FP4_SCALE, get_quantization_fns, get_quantization_range, cast_to_eBm0
from ..helpers import split_dim

# Utility function for inversion.
def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


class Quantizer:

    def __init__(
        self, 
        bits: int, 
        symmetric: bool = True,
        format: str = "int",
        granularity: str = "channel",
        observer: str = "minmax",
        dim: int = -1,
        group_size: Optional[int] = None,
        scale_precision: str = "fp16",
        scale_min_clip: Optional[float] = None
    ):
        # Sanity checks
        if format in ["fp", "nvfp", "mxfp"]:
            assert symmetric, "Only symmetric quantization is supported for floating point formats."

        if granularity == "group":
            assert group_size is not None, "Group size must be specified when granularity is 'group'."
        else:
            assert group_size is None, "Group size must be None when granularity is not 'group'."

        self.bits = bits
        self.symmetric = symmetric
        self.format = QuantizationFormat(format)
        self.granularity = QuantizationGranularity(granularity)
        self.observer = QuantizationObserver(observer)
        self.scale_precision = ScalePrecision(scale_precision)
        self.dim = dim
        self.group_size = group_size
        self.scale_min_clip = scale_min_clip

        self.quant_fn, self.dequant_fn, self.quant_dequant_fn = get_quantization_fns(
            format=self.format,
            bits=self.bits,
        )

        self.q_min, self.q_max = get_quantization_range(
            format=self.format,
            bits=self.bits,
            symmetric=self.symmetric,
        )
        
        # Global scale is 3 for MXFP quantization
        if self.format == QuantizationFormat.MXFP:
            self.global_scale = torch.tensor([3.0], dtype=torch.float32)
        else:
            self.global_scale = torch.tensor([float("inf")], dtype=torch.float32)
        # Scale tracking is needed only for E4M3 scale quantization
        self._track_global_scale = (self.scale_precision == ScalePrecision.E4M3)

    def _reshape_before_quantization(
        self, 
        x: torch.Tensor, 
        scales: Optional[torch.Tensor] = None,
        zeros: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.group_size:
            dim = x.ndim - 1 if self.dim == -1 else self.dim
            num_groups = x.shape[dim] // self.group_size
            x = split_dim(x, num_groups, dim)
            if scales is not None:
                scales = scales.unsqueeze(dim + 1)
            if zeros is not None:
                zeros = zeros.unsqueeze(dim + 1)
        return x, scales, zeros

    def get_quantization_params(
        self, 
        x: torch.Tensor,
        # MSE observer quantization params
        scale_search_iters: int = 100,
        max_scale_shrink_factor: float = 0.80,
        error_norm: float = 2.4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get scale and zero point for an input tensor.
        """
        dim = x.ndim - 1 if self.dim == -1 else self.dim
        if self.granularity == QuantizationGranularity.GROUP:
            reduce_dim = dim + 1
        elif self.granularity == QuantizationGranularity.CHANNEL:
            reduce_dim = dim
        else:
            reduce_dim = None
        x, _, _ = self._reshape_before_quantization(x)

        x_min = x.amin(dim=reduce_dim, keepdim=True)
        x_max = x.amax(dim=reduce_dim, keepdim=True)

        if self.symmetric:
            scales = 2 * torch.maximum(-x_min, x_max) / (self.q_max - self.q_min)
            zeros =  torch.zeros_like(x_min)
        else:
            scales = (x_max - x_min) / (self.q_max - self.q_min)
            zeros = -(x_min / scales).round()

        if self.observer == QuantizationObserver.MSE:
            init_scales = scales.clone() 
            best_quantization_error = torch.full(x.shape[:-1], float("inf"), device=x.device, dtype=x.dtype)

            for i in range(scale_search_iters):
                scale_shrink_factor = 1 - i * max_scale_shrink_factor / scale_search_iters
                candidate_scales = scale_shrink_factor * init_scales
                candidate_zeros = torch.zeros_like(x_min) if self.symmetric else -(x_min / candidate_scales).round() 
                q = self.quant_fn(x, candidate_scales, candidate_zeros, self.q_min, self.q_max)
                x_reconstructed = self.dequant_fn(q, candidate_scales, candidate_zeros)
                quantization_error = (x - x_reconstructed).abs_().pow_(error_norm).sum(dim=-1)

                if (quantization_error < best_quantization_error).any():
                    improved_ids = torch.where(quantization_error < best_quantization_error)
                    best_quantization_error[improved_ids] = quantization_error[improved_ids]
                    scales[improved_ids] = candidate_scales[improved_ids]
                    if not self.symmetric:
                        zeros[improved_ids] = candidate_zeros[improved_ids]

        # Reshape back
        if self.group_size:
            x = x.flatten(dim, dim + 1)
            scales = scales.squeeze(dim + 1)
            if zeros is not None:
                zeros = zeros.squeeze(dim + 1)

        if self.scale_precision == ScalePrecision.E4M3:
            with torch.no_grad():
                if self._track_global_scale:
                    current_global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(x.abs().max().to(torch.float32).view(1))
                    if not current_global_scale:
                        raise ValueError(f"Current global scale is not finite: {current_global_scale}\n")
                    # Update global scale using min of current and computed scale
                    self.global_scale = torch.minimum(self.global_scale.to(x.device), current_global_scale)
                    
                    if not self.global_scale.isfinite():
                        raise ValueError(f"Global scale is not finite: {self.global_scale}\n")
                    
                # Clamp, convert to fp8, convert back, and rescale in one chain
                scales = (scales * self.global_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX) \
                    .to(torch.float8_e4m3fn) \
                    .to(torch.float32) \
                    .mul(get_reciprocal(self.global_scale)) \
                    .to(x.dtype)
        
        elif self.scale_precision == ScalePrecision.E8M0:
            # Inspired by quantize_tseng (see https://github.com/IST-DASLab/Quartet/blob/main/notebooks/benchmark_mxfp4.ipynb)
            # NOTE (in quartet x.abs().max() is defined as a scale insteaf of x.abs().max() / q_max )
            scales = cast_to_eBm0(FP4_E2M1_MAX * scales, ebits=8, emax=2) / FP4_SCALE

        # Set scales to 1 if zero
        scales[scales == 0] = 1

        if scales.isnan().any():
            raise ValueError(f"Scales are not finite.")
      
        return scales, zeros
        
    def quantize(self, x: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = x.shape
        q = self.quant_fn(
            *self._reshape_before_quantization(x, scales, zeros), 
            self.q_min, 
            self.q_max
        ).reshape(original_shape)
        return q

    def dequantize(self, q: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = q.shape
        return self.dequant_fn(
            *self._reshape_before_quantization(q, scales, zeros), 
        ).reshape(original_shape)
    
    def __call__(self, x: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = x.shape
        q = self.quant_dequant_fn(
            *self._reshape_before_quantization(x, scales, zeros), 
            self.q_min, 
            self.q_max
        ).reshape(original_shape)
        return q
