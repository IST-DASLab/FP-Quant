from typing import Optional, Tuple

import torch
from torch import nn
from torch.autograd import Function


from ..utils import FPQuantDtype
from .qutlass_ops import (
    # Forward quantization
    fused_quantize_mx_op,
    fused_quantize_nv_op,
    # GEMMs
    matmul_ada_mxf4_bf16_tn_op,
    matmul_mxf4_bf16_tn_op,
    matmul_nvf4_bf16_tn_op,
    matmul_mxf8_bf16_tn_op,
    matmul_mxf8_bf16_nn_op,
    # Backward quantization
    backward_t_bf16_op,
    backward_qt_bf16_op,
    backward_bf16_square_double_mxfp8_op,
    mxfp4_transpose_mxfp8_op,
)


def forward_quantize(
    x: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: FPQuantDtype,
    forward_method: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype == FPQuantDtype.MXFP4:
        qweight, scales, mask = fused_quantize_mx_op(
            x.to(torch.bfloat16),
            hadamard_matrix.to(torch.bfloat16),
            forward_method,
            forward_method == "quest" and x.requires_grad,
        )
        return qweight, scales, mask
    elif dtype == FPQuantDtype.NVFP4:
        qweight, scales = fused_quantize_nv_op(
            x.to(torch.bfloat16),
            hadamard_matrix.to(torch.bfloat16),
            global_scale.float(),
        )
        return qweight, scales, None  # TODO: add mask
    else:
        raise ValueError(f"Unsupported forward dtype: {dtype}")


def forward_gemm(x_q, w_q, x_scales, w_scales, alpha, dtype: FPQuantDtype):
    if dtype == FPQuantDtype.MXFP4:
        if False and x_q.shape[0] <= 64:  # TODO: remove when ada alpha is fixed
            return matmul_ada_mxf4_bf16_tn_op(
                x_q, w_q, x_scales, w_scales, alpha.float()
            )
        else:
            return matmul_mxf4_bf16_tn_op(x_q, w_q, x_scales, w_scales, alpha.float())
    elif dtype == FPQuantDtype.NVFP4:
        return matmul_nvf4_bf16_tn_op(x_q, w_q, x_scales, w_scales, alpha.float())
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


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


class FPQuant4x4MasterFn(Function):
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
        ctx.forward_method = forward_method
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

        backward_hadamard_matrix = forward_hadamard_matrix.to(torch.bfloat16) * (
            torch.randint(
                0, 2, (32,), device=forward_hadamard_matrix.device, dtype=torch.bfloat16
            )
            * 2.0
            - 1.0
        )

        grad_output_hb_e2m1, grad_output_hb_e8m0, _ = fused_quantize_mx_op(
            grad_output.flatten(end_dim=-2),
            backward_hadamard_matrix,
            "abs_max",
            False,
        )

        hft_weightt_hb_e2m1, hft_weightt_hb_e8m0 = backward_qt_bf16_op(
            weight_q,
            weight_scales,
            backward_hadamard_matrix,
            weight_global_scale,
        )

        grad_input_hf = matmul_mxf4_bf16_tn_op(
            grad_output_hb_e2m1.view(torch.uint8),
            hft_weightt_hb_e2m1.view(torch.uint8),
            grad_output_hb_e8m0,
            hft_weightt_hb_e8m0,
            act_global_scale / act_global_scale / 9.0,
        )

        if x_flat_mask is not None:
            grad_input_hf *= (
                _unpack_mask(x_flat_mask).view_as(grad_input_hf).to(grad_input_hf.dtype)
            )
        grad_input = (
            grad_input_hf.view(-1, 32) @ forward_hadamard_matrix.T.to(torch.bfloat16)
        ).view(ctx.x_shape)

        grad_outputt_hb_e2m1, grad_outputt_hb_e8m0 = backward_t_bf16_op(
            grad_output.flatten(end_dim=-2),
            backward_hadamard_matrix,
        )
        hft_inputt_hb_e2m1, hft_inputt_hb_e8m0 = backward_qt_bf16_op(
            x_flat_q,
            x_flat_scales,
            backward_hadamard_matrix,
            act_global_scale,
        )
        grad_weight_hf = matmul_mxf4_bf16_tn_op(
            grad_outputt_hb_e2m1.view(torch.uint8),
            hft_inputt_hb_e2m1.view(torch.uint8),
            grad_outputt_hb_e8m0,
            hft_inputt_hb_e8m0,
            act_global_scale / act_global_scale / 9.0,
        )

        if weight_mask is not None:
            grad_weight_hf *= (
                _unpack_mask(weight_mask)
                .view_as(grad_weight_hf)
                .to(grad_weight_hf.dtype)
            )
        grad_weight = (
            grad_weight_hf.view(-1, 32) @ forward_hadamard_matrix.T.to(torch.bfloat16)
        ).view(grad_output.size(-1), weight_q.size(-1) * 2)

        grad_bias = (
            grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None
        )

        return grad_input, grad_weight, None, None, grad_bias, None, None, None


class FPQuant4x8MasterFn(Function):
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

        x_flat_t_fp8, x_flat_t_scales = mxfp4_transpose_mxfp8_op(
            x_flat_q,
            x_flat_scales,
        )
        weight_t_fp8, weight_t_scales = mxfp4_transpose_mxfp8_op(
            weight_q,
            weight_scales,
        )

        grad_output = grad_output.flatten(end_dim=-2)
        grad_output_fp8, grad_output_hid_scales, grad_output_batch_scales = (
            backward_bf16_square_double_mxfp8_op(
                grad_output,
            )
        )

        grad_input = matmul_mxf8_bf16_tn_op(
            grad_output_fp8,
            weight_t_fp8,
            grad_output_hid_scales,
            weight_t_scales,
            1 / weight_global_scale.float(),
        )[: grad_output.shape[0], :]

        if x_flat_mask is not None:
            grad_input *= (
                _unpack_mask(x_flat_mask).view_as(grad_input).to(grad_input.dtype)
            )
        grad_input = (
            grad_input.view(-1, forward_hadamard_matrix.shape[1])
            @ forward_hadamard_matrix.T.to(torch.bfloat16)
        ).view(ctx.x_shape)

        grad_weight = matmul_mxf8_bf16_nn_op(
            grad_output_fp8,
            x_flat_t_fp8,
            grad_output_batch_scales,
            x_flat_t_scales,
            1 / act_global_scale.float(),
        )

        if weight_mask is not None:
            grad_weight *= (
                _unpack_mask(weight_mask).view_as(grad_weight).to(grad_weight.dtype)
            )
        grad_weight = (
            grad_weight.view(-1, forward_hadamard_matrix.shape[1])
            @ forward_hadamard_matrix.T.to(torch.bfloat16)
        ).view(grad_output.size(-1), weight_q.size(-1) * 2)

        grad_bias = (
            grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None
        )

        return grad_input, grad_weight, None, None, grad_bias, None, None, None


class FPQuant4x8NoMasterFn(Function):
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
            x_flat,
            weight,
        )

        return y

    @staticmethod
    @torch.compile(mode="max-autotune", fullgraph=True)
    def backward(ctx, grad_output: torch.Tensor):
        (
            x_flat,
            weight,
        ) = ctx.saved_tensors

        grad_output = grad_output.flatten(end_dim=-2)

        grad_input = torch.einsum(
            "...i,ij->...j",
            grad_output,
            weight,
        ).view(ctx.x_shape)

        grad_weight = torch.einsum(
            "...i,...j->ij",
            grad_output,
            x_flat,
        ).view(grad_output.size(-1), weight.size(-1))

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
