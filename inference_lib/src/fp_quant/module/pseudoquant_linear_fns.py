from typing import Optional

import torch
from torch import nn
from torch.autograd import Function

from ..utils import FPQuantDtype
from .triton.mxfp4 import mxfp4_forward_kernel_wrapper
from .triton.nvfp4 import nvfp4_forward_kernel_wrapper


def forward_pseudoquantize(
    x: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    dtype: FPQuantDtype,
    forward_method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    match dtype:
        case FPQuantDtype.MXFP4:
            if forward_method == "quest":
                gaussian_scale = 2.92247856 / 6.0
                quest = True
            elif forward_method == "abs_max":
                gaussian_scale = 3.0 / 4.0
                quest = False
            else:
                raise ValueError(f"Unsupported forward method: {forward_method}")

            x_dequantized, mask = mxfp4_forward_kernel_wrapper(
                x,
                hadamard_matrix,
                return_clip_mask=True,
                stochastic_round=False,
                quest=quest,
                gaussian_scale=gaussian_scale,
            )
            return x_dequantized, mask
        case FPQuantDtype.NVFP4:
            assert forward_method == "abs_max", "NVFP4 only supports abs_max method"
            x_dequantized = nvfp4_forward_kernel_wrapper(
                x,
                hadamard_matrix,
                global_scale,
            )
            return x_dequantized, torch.ones_like(x_dequantized, dtype=torch.bool)
        case FPQuantDtype.MXFP8:
            raise NotImplementedError(
                "MXFP8 is not supported for forward quantization yet"
            )
        case _:
            raise ValueError(f"Unsupported forward dtype: {dtype}")


class PseudoQuant4x16MasterFn(Function):
    @staticmethod
    # @torch.compile()
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        forward_hadamard_matrix: torch.Tensor,
        global_scale: Optional[torch.Tensor],
        dtype: FPQuantDtype,
        forward_method: str,
    ):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Pseudoquantize input
        x_flat_dq, x_flat_mask = forward_pseudoquantize(
            x_flat, forward_hadamard_matrix, global_scale, dtype, forward_method
        )

        # Pseudoquantize weights
        weight_dq, weight_mask = forward_pseudoquantize(
            weight, forward_hadamard_matrix, global_scale, dtype, forward_method
        )

        y = torch.nn.functional.linear(x_flat_dq, weight_dq, bias)

        y = y.unflatten(dim=0, sizes=x.shape[:-1])

        ctx.x_shape = x.shape
        ctx.dtype = dtype
        ctx.bias_present = bias is not None
        ctx.save_for_backward(
            x_flat_dq,
            weight_dq,
            x_flat_mask,
            weight_mask,
            forward_hadamard_matrix,
        )

        return y

    @staticmethod
    # @torch.compile()
    def backward(ctx, grad_output: torch.Tensor):
        (
            x_flat_dq,
            weight_dq,
            x_flat_mask,
            weight_mask,
            forward_hadamard_matrix,
        ) = ctx.saved_tensors

        grad_output_flat = grad_output.flatten(end_dim=-2)

        grad_input = torch.einsum("...j,ji->...i", grad_output_flat, weight_dq)
        grad_input = (
            (grad_input.view(-1, 32) * x_flat_mask.view(-1, 32).to(grad_input.dtype))
            @ forward_hadamard_matrix.T
        ).view(ctx.x_shape)

        grad_weight = torch.einsum("...j,...i->ji", grad_output_flat, x_flat_dq)
        grad_weight = (
            (grad_weight.view(-1, 32) * weight_mask.view(-1, 32).to(grad_weight.dtype))
            @ forward_hadamard_matrix.T
        ).view(grad_output.size(-1), weight_dq.size(-1))

        grad_bias = (
            grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None
        )

        return grad_input, grad_weight, grad_bias, None, None, None


class PseudoQuant4x16NoMasterFn(Function):
    @staticmethod
    # @torch.compile()
    def forward(
        ctx,
        x: torch.Tensor,
        weight_dq: torch.Tensor,
        bias: Optional[torch.Tensor],
        forward_hadamard_matrix: torch.Tensor,
        global_scale: Optional[torch.Tensor],
        dtype: FPQuantDtype,
        forward_method: str,
    ):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Pseudoquantize input
        x_flat_dq, x_flat_mask = forward_pseudoquantize(
            x_flat, forward_hadamard_matrix, global_scale, dtype, forward_method
        )

        y = torch.nn.functional.linear(x_flat_dq, weight_dq, bias)

        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if bias is not None:
            y += bias

        ctx.x_shape = x.shape
        ctx.dtype = dtype
        ctx.bias_present = bias is not None
        ctx.save_for_backward(
            x_flat_dq,
            weight_dq,
            x_flat_mask,
            forward_hadamard_matrix,
        )

        return y

    @staticmethod
    # @torch.compile()
    def backward(ctx, grad_output: torch.Tensor):
        _, weight_dq, x_flat_mask, forward_hadamard_matrix = ctx.saved_tensors

        grad_output_flat = grad_output.flatten(end_dim=-2)

        grad_input = torch.einsum("...j,ji->...i", grad_output_flat, weight_dq)
        x_flat_mask = _unpack_mask(x_flat_mask)
        grad_input = (
            (grad_input.view(-1, 32) * x_flat_mask.view(-1, 32).to(grad_input.dtype))
            @ forward_hadamard_matrix.T
        ).view(ctx.x_shape)

        grad_bias = (
            grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None
        )

        return grad_input, None, None, grad_bias, None, None, None
