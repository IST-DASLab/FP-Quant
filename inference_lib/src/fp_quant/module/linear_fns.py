from typing import Optional, Tuple

import torch
from torch import nn
from torch.autograd import Function

try:
    from qutlass import (
        fusedQuantizeMx,
        fusedQuantizeNv,
        matmul_ada_mxf4_bf16_tn,
        matmul_mxf4_bf16_tn,
        matmul_nvf4_bf16_tn,
    )
    from qutlass.utils import to_blocked

    HAS_QUTLASS = True
except ImportError:
    HAS_QUTLASS = False

from ..utils import FPQuantDtype


@torch.library.custom_op("fp_quant::fused_quantize_mx_op", mutates_args=())
def fused_quantize_mx_op(
    x_flat: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    forward_method: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fusedQuantizeMx(
        x_flat.to(torch.bfloat16),
        hadamard_matrix.to(torch.bfloat16),
        method=forward_method,
    )


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
    x: torch.Tensor,
    w: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return matmul_mxf4_bf16_tn(
        x, w, to_blocked(xs), to_blocked(ws).view(torch.float8_e8m0fnu), alpha.float()
    )


@matmul_mxf4_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16)


@torch.library.custom_op("fp_quant::matmul_ada_mxf4_bf16_tn_op", mutates_args=())
def matmul_ada_mxf4_bf16_tn_op(
    x: torch.Tensor,
    w: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return matmul_ada_mxf4_bf16_tn(
        x, w, xs, ws.view(torch.float8_e8m0fnu), alpha.float()
    )


@matmul_ada_mxf4_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16)


@torch.library.custom_op("fp_quant::fused_quantize_nv_op", mutates_args=())
def fused_quantize_nv_op(
    x_flat: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    global_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fusedQuantizeNv(
        x_flat.to(torch.bfloat16),
        hadamard_matrix.to(torch.bfloat16),
        global_scale.float(),
    )


@fused_quantize_nv_op.register_fake
def _(x_flat, hadamard_matrix, global_scale):
    xh_e2m1 = torch.empty(
        *x_flat.shape[:-1],
        x_flat.size(-1) // 2,
        dtype=torch.uint8,
        device=x_flat.device,
    )

    rows, cols = x_flat.numel() // x_flat.size(-1), x_flat.size(-1) // 16
    n_row_blocks = (rows + 128 - 1) // 128
    n_col_blocks = (cols + 4 - 1) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    xh_e4m3 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e4m3fn, device=x_flat.device
    )

    return xh_e2m1, xh_e4m3


@torch.library.custom_op("fp_quant::matmul_nvf4_bf16_tn_op", mutates_args=())
def matmul_nvf4_bf16_tn_op(
    x: torch.Tensor,
    w: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return matmul_nvf4_bf16_tn(
        x, w, to_blocked(xs), to_blocked(ws.view(torch.float8_e4m3fn)), alpha.float()
    )


@matmul_nvf4_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16)


def forward_quantize(
    x: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: FPQuantDtype,
    forward_method: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype == FPQuantDtype.MXFP4:
        qweight, scales = fused_quantize_mx_op(
            x,
            hadamard_matrix,
            forward_method,
        )
        return qweight, scales, None  # TODO: add mask
    elif dtype == FPQuantDtype.NVFP4:
        qweight, scales = fused_quantize_nv_op(x, hadamard_matrix, global_scale)
        return qweight, scales, None  # TODO: add mask
    else:
        raise ValueError(f"Unsupported forward dtype: {dtype}")


def forward_gemm(x_q, w_q, x_scales, w_scales, alpha, dtype: FPQuantDtype):
    if dtype == FPQuantDtype.MXFP4:
        if False and x_q.shape[0] <= 64:  # TODO: remove when ada alpha is fixed
            return matmul_ada_mxf4_bf16_tn_op(x_q, w_q, x_scales, w_scales, alpha)
        else:
            return matmul_mxf4_bf16_tn_op(x_q, w_q, x_scales, w_scales, alpha)
    elif dtype == FPQuantDtype.NVFP4:
        return matmul_nvf4_bf16_tn_op(x_q, w_q, x_scales, w_scales, alpha)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


GRID_FP4 = torch.tensor(
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


def _dq_fp4(x_e2m1: torch.Tensor, scales: torch.Tensor, alpha, dtype: FPQuantDtype):
    global GRID_FP4
    device = x_e2m1.device
    result_shape = x_e2m1.shape[:-1] + (x_e2m1.shape[-1] * 2,)

    GRID_FP4 = GRID_FP4.to(device)

    xq_unpacked = torch.stack([x_e2m1 & 0xF, x_e2m1 >> 4], dim=-1).to(torch.int32)
    x_fp4_dq = GRID_FP4[xq_unpacked]
    if dtype == FPQuantDtype.MXFP4:
        scales_dq = scales.view(torch.float8_e8m0fnu).to(torch.bfloat16)
        gs = 32
    elif dtype == FPQuantDtype.NVFP4:
        scales_dq = scales.view(torch.float8_e4m3fn).to(torch.bfloat16)
        gs = 16
    else:
        raise ValueError(f"Unsupported FPQuant dtype {dtype}")

    x_dq = (x_fp4_dq.view(-1, gs) * scales_dq.view(-1, 1)).view(
        *result_shape
    ) / alpha.to(torch.bfloat16)
    return x_dq


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


class FPQuant4x16MasterFn(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_global_scale: torch.Tensor,
        act_global_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        forward_hadamard_matrix: torch.Tensor,
        dtype: FPQuantDtype,
        forward_method: str,
    ):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_scales, x_flat_mask = forward_quantize(
            x_flat, forward_hadamard_matrix, act_global_scale, dtype, forward_method
        )

        # Quantize weights
        weight_q, weight_scales, weight_mask = forward_quantize(
            weight, forward_hadamard_matrix, weight_global_scale, dtype, forward_method
        )

        y = forward_gemm(
            x_flat_q,
            weight_q,
            x_flat_scales,
            weight_scales,
            1.0 / (weight_global_scale * act_global_scale),
            dtype,
        )

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
            act_global_scale,
            weight_global_scale,
            x_flat_mask,
            weight_mask,
            forward_hadamard_matrix,
        )

        return y

    @staticmethod
    @torch.compile(mode="max-autotune", fullgraph=True)
    def backward(ctx, grad_output: torch.Tensor):
        (
            x_flat_q,
            weight_q,
            x_flat_scales,
            weight_scales,
            act_global_scale,
            weight_global_scale,
            x_flat_mask,
            weight_mask,
            forward_hadamard_matrix,
        ) = ctx.saved_tensors

        x_flat_dequantized = _dq_fp4(
            x_flat_q, x_flat_scales, act_global_scale, ctx.dtype
        )
        weight_dequantized = _dq_fp4(
            weight_q, weight_scales, weight_global_scale, ctx.dtype
        )
        grad_output_flat = grad_output.flatten(end_dim=-2)

        grad_input = torch.einsum("...j,ji->...i", grad_output_flat, weight_dequantized)
        if x_flat_mask is not None:
            grad_input *= (
                _unpack_mask(x_flat_mask).view_as(grad_input).to(grad_input.dtype)
            )
        grad_input = (
            grad_input.view(-1, 32) @ forward_hadamard_matrix.T.to(torch.bfloat16)
        ).view(ctx.x_shape)

        grad_weight = torch.einsum(
            "...j,...i->ji", grad_output_flat, x_flat_dequantized
        )
        if weight_mask is not None:
            grad_weight *= (
                _unpack_mask(weight_mask).view_as(grad_weight).to(grad_weight.dtype)
            )
        grad_weight = (
            grad_weight.view(-1, 32) @ forward_hadamard_matrix.T.to(torch.bfloat16)
        ).view(grad_output.size(-1), weight_q.size(-1) * 2)

        grad_bias = (
            grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None
        )

        return grad_input, grad_weight, None, None, grad_bias, None, None, None


class FPQuant4x16NoMasterFn(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight_q: torch.Tensor,
        weight_scales: torch.Tensor,
        weight_global_scale: torch.Tensor,
        act_global_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        forward_hadamard_matrix: torch.Tensor,
        dtype: FPQuantDtype,
        forward_method: str,
    ):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_scales, x_flat_mask = forward_quantize(
            x_flat, forward_hadamard_matrix, act_global_scale, dtype, forward_method
        )

        y = forward_gemm(
            x_flat_q,
            weight_q,
            x_flat_scales,
            weight_scales,
            1.0 / (weight_global_scale * act_global_scale),
            dtype,
        )

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
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError(
            "Backward pass is not implemented for FPQuant4x16NoMasterFn yet"
        )
