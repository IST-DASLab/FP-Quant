from typing import Optional

import torch
from torch import nn
from torch.autograd import Function

try:
    from qutlass import fusedQuantizeMx, matmul_ada_mxf4_bf16_tn, matmul_mxf4_bf16_tn
    from qutlass.utils import to_blocked

    HAS_QUTLASS = True
except ImportError:
    HAS_QUTLASS = False

from ..utils import FPQuantDtype


@torch.library.custom_op("fp_quant::fused_quantize_op", mutates_args=())
def fused_quantize_mx_op(
    x_flat: torch.Tensor, hadamard_matrix: torch.Tensor, forward_method: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fusedQuantizeMx(x_flat, hadamard_matrix, method=forward_method)


@fused_quantize_mx_op.register_fake
def _(x_flat, hadamard_matrix, forward_method):
    rows, cols = x_flat.size(0), x_flat.size(1) // 32
    padded_rows = ((rows + 128 - 1) // 128) * 128
    padded_cols = ((cols + 4 - 1) // 4) * 4

    xh_e2m1 = torch.empty(
        x_flat.size(0), x_flat.size(1) // 2, dtype=torch.uint8, device=x_flat.device
    )
    xh_e8m0 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e8m0fnu, device=x_flat.device
    )

    return xh_e2m1, xh_e8m0


@torch.library.custom_op("fp_quant::matmul_mxf4_bf16_tn_op", mutates_args=())
def matmul_mxf4_bf16_tn_op(
    x: torch.Tensor, w: torch.Tensor, xs: torch.Tensor, ws: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    return matmul_mxf4_bf16_tn(
        x, w, to_blocked(xs), to_blocked(ws).view(torch.float8_e8m0fnu), alpha
    )


@matmul_mxf4_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16)


@torch.library.custom_op("fp_quant::matmul_ada_mxf4_bf16_tn_op", mutates_args=())
def matmul_ada_mxf4_bf16_tn_op(
    x: torch.Tensor, w: torch.Tensor, xs: torch.Tensor, ws: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    return matmul_ada_mxf4_bf16_tn(x, w, xs, ws.view(torch.float8_e8m0fnu), alpha)


@matmul_ada_mxf4_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16)


FP4_GRID = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.bfloat16,
)


def dequantize_mxf4_bf16(xq: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    xq_unpacked = torch.stack([xq & 0xF, xq >> 4], dim=-1).to(torch.int32)
    x_dequantized = FP4_GRID.to(xq.device)[xq_unpacked]

    float_exps = 2 ** (exps.view(dtype=torch.uint8).to(torch.int32) - 127).to(
        torch.bfloat16
    )
    return (x_dequantized.view(-1, 32) * float_exps.view(-1, 1)).view(
        *xq.shape[:-1], -1
    )


# @torch.compile()
@torch.inference_mode()
def _unpack_mask(clip_mask: torch.Tensor) -> torch.Tensor:
    clip_mask_unpacked_dq = torch.zeros(
        *clip_mask.shape[:-1],
        clip_mask.size(-1) * 8,
        dtype=torch.bool,
        device=clip_mask.device,
    )
    for i in range(8):
        clip_mask_unpacked_dq[..., i::8] = (clip_mask >> i) & 1
    return clip_mask_unpacked_dq


def forward_quantize(
    x: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    dtype: FPQuantDtype,
    forward_method: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    match dtype:
        case FPQuantDtype.MXFP4:
            qweight_scales, scales = fused_quantize_mx_op(
                x, hadamard_matrix, forward_method
            )
            return qweight_scales, scales, None  # TODO: add mask
        case FPQuantDtype.MXFP8:
            raise NotImplementedError(
                "MXFP8 is not supported for forward quantization yet"
            )
        case _:
            raise ValueError(f"Unsupported forward dtype: {dtype}")


def forward_gemm(x_q, w_q, x_scales, w_scales, alpha):
    if isinstance(alpha, float):
        alpha = torch.tensor(alpha, dtype=torch.float32)
    if x_q.shape[0] <= 64:
        return matmul_ada_mxf4_bf16_tn_op(x_q, w_q, x_scales, w_scales, alpha)
    else:
        return matmul_mxf4_bf16_tn_op(x_q, w_q, x_scales, w_scales, alpha)


class FPQuant4x16MasterFn(Function):
    @staticmethod
    # @torch.compile()
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        forward_hadamard_matrix: torch.Tensor,
        dtype: FPQuantDtype,
        forward_method: str,
    ):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_scales, x_flat_mask = forward_quantize(
            x_flat, forward_hadamard_matrix, dtype, forward_method
        )

        # Quantize weights
        weight_q, weight_scales, weight_mask = forward_quantize(
            weight, forward_hadamard_matrix, dtype, forward_method
        )

        y = forward_gemm(x_flat_q, weight_q, x_flat_scales, weight_scales, 1.0 / 9.0)

        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if bias is not None:
            y += bias

        ctx.x_shape = x.shape
        ctx.dtype = dtype
        ctx.bias_present = bias is not None
        ctx.save_for_backward(
            x_flat_q,
            weight_q,
            x_flat_scales,
            weight_scales,
            x_flat_mask,
            weight_mask,
            forward_hadamard_matrix,
        )

        return y

    @staticmethod
    # @torch.compile()
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError(
            "Backward pass is not implemented for FPQuant4x16MasterFn yet"
        )
        (
            x_flat_q,
            weight_q,
            x_flat_scales,
            weight_scales,
            x_flat_mask,
            weight_mask,
            forward_hadamard_matrix,
        ) = ctx.saved_tensors

        x_flat_dequantized = dequantize_mxf4_bf16(x_flat_q, x_flat_scales) / 3.0
        weight_dequantized = dequantize_mxf4_bf16(weight_q, weight_scales) / 3.0
        grad_output_flat = grad_output.flatten(end_dim=-2)

        grad_input = torch.einsum("...j,ji->...i", grad_output_flat, weight_dequantized)
        x_flat_mask = _unpack_mask(x_flat_mask)
        grad_input = (
            (grad_input.view(-1, 32) * x_flat_mask.view(-1, 32).to(grad_input.dtype))
            @ forward_hadamard_matrix.T
        ).view(ctx.x_shape)

        grad_weight = torch.einsum(
            "...j,...i->ji", grad_output_flat, x_flat_dequantized
        )
        weight_mask = _unpack_mask(weight_mask)
        grad_weight = (
            (grad_weight.view(-1, 32) * weight_mask.view(-1, 32).to(grad_weight.dtype))
            @ forward_hadamard_matrix.T
        ).view(grad_output.size(-1), weight_q.size(-1) * 2)

        grad_bias = (
            grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None
        )

        return grad_input, grad_weight, grad_bias, None, None, None


class FPQuant4x16NoMasterFn(Function):
    @staticmethod
    # @torch.compile()
    def forward(
        ctx,
        x: torch.Tensor,
        weight_q: torch.Tensor,
        weight_scales: torch.Tensor,
        bias: Optional[torch.Tensor],
        forward_hadamard_matrix: torch.Tensor,
        dtype: FPQuantDtype,
        forward_method: str,
    ):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_scales, x_flat_mask = forward_quantize(
            x_flat, forward_hadamard_matrix, dtype, forward_method
        )

        y = forward_gemm(x_flat_q, weight_q, x_flat_scales, weight_scales, 1.0 / 9.0)

        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if bias is not None:
            y += bias

        ctx.x_shape = x.shape
        ctx.dtype = dtype
        ctx.bias_present = bias is not None
        ctx.save_for_backward(
            x_flat_q,
            weight_q,
            x_flat_scales,
            weight_scales,
            x_flat_mask,
            forward_hadamard_matrix,
        )

        return y

    @staticmethod
    # @torch.compile()
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError(
            "Backward pass is not implemented for FPQuant4x16NoMasterFn yet"
        )
        _, weight_q, _, weight_scales, x_flat_mask, forward_hadamard_matrix = (
            ctx.saved_tensors
        )

        weight_dequantized = dequantize_mxf4_bf16(weight_q, weight_scales) / 3.0
        grad_output_flat = grad_output.flatten(end_dim=-2)

        grad_input = torch.einsum("...j,ji->...i", grad_output_flat, weight_dequantized)
        x_flat_mask = _unpack_mask(x_flat_mask)
        grad_input = (
            (grad_input.view(-1, 32) * x_flat_mask.view(-1, 32).to(grad_input.dtype))
            @ forward_hadamard_matrix.T
        ).view(ctx.x_shape)

        grad_bias = (
            grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None
        )

        return grad_input, None, None, grad_bias, None, None, None
