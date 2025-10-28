from typing import Tuple

import torch

try:
    from qutlass import (
        # Forward quantization
        fusedQuantizeMx,
        fusedQuantizeNv,
        # GEMMs
        matmul_ada_mxf4_bf16_tn,
        matmul_mxf4_bf16_tn,
        matmul_nvf4_bf16_tn,
        matmul_mxf8_bf16_tn,
        matmul_mxf8_bf16_nn,
        # Backward quantization
        backward_t_bf16,
        backward_qt_bf16,
        backward_bf16_square_double_mxfp8,
        mxfp4_transpose_mxfp8,
    )

    # Layout
    from qutlass.utils import to_blocked as to_blocked_qutlass

    HAS_QUTLASS = True
except ImportError:
    HAS_QUTLASS = False

from ..utils import FPQuantDtype


### Forward quantization


@torch.library.custom_op("fp_quant::fused_quantize_mx_op", mutates_args=())
def fused_quantize_mx_op(
    x_flat: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    forward_method: str,
    return_mask: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = fusedQuantizeMx(
        x_flat,
        hadamard_matrix,
        method=forward_method,
        return_mask=return_mask,
    )
    if not return_mask:
        tensors = tensors + (None,)
    return tensors


@fused_quantize_mx_op.register_fake
def _(x_flat, hadamard_matrix, forward_method, return_mask):
    rows, cols = x_flat.size(0), x_flat.size(1) // 32
    padded_rows = ((rows + 128 - 1) // 128) * 128
    padded_cols = ((cols + 4 - 1) // 4) * 4

    xh_e2m1 = torch.empty(
        x_flat.size(0), x_flat.size(1) // 2, dtype=torch.uint8, device=x_flat.device
    )
    xh_e8m0 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e8m0fnu, device=x_flat.device
    )
    if return_mask:
        mask = torch.empty(rows, cols * 4, dtype=torch.uint8, device=x_flat.device)
    else:
        mask = None

    return xh_e2m1, xh_e8m0, mask


@torch.library.custom_op("fp_quant::fused_quantize_nv_op", mutates_args=())
def fused_quantize_nv_op(
    x_flat: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    global_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fusedQuantizeNv(
        x_flat,
        hadamard_matrix,
        global_scale,
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


### GEMMs


@torch.library.custom_op("fp_quant::matmul_mxf4_bf16_tn_op", mutates_args=())
def matmul_mxf4_bf16_tn_op(
    x: torch.Tensor,
    w: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return matmul_mxf4_bf16_tn(
        x,
        w,
        to_blocked_qutlass(xs, use_triton_kernel=True),
        to_blocked_qutlass(ws, use_triton_kernel=True).view(torch.float8_e8m0fnu),
        alpha,
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
    return matmul_ada_mxf4_bf16_tn(x, w, xs, ws.view(torch.float8_e8m0fnu), alpha)


@matmul_ada_mxf4_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16)


@torch.library.custom_op("fp_quant::matmul_nvf4_bf16_tn_op", mutates_args=())
def matmul_nvf4_bf16_tn_op(
    x: torch.Tensor,
    w: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return matmul_nvf4_bf16_tn(
        x,
        w,
        to_blocked_qutlass(xs, use_triton_kernel=True),
        to_blocked_qutlass(ws.view(torch.float8_e4m3fn), use_triton_kernel=True),
        alpha,
    )


@matmul_nvf4_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16)


@torch.library.custom_op("fp_quant::matmul_mxf8_bf16_tn_op", mutates_args=())
def matmul_mxf8_bf16_tn_op(
    x: torch.Tensor,
    w: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return matmul_mxf8_bf16_tn(
        x,
        w,
        to_blocked_qutlass(xs, use_triton_kernel=True),
        to_blocked_qutlass(ws, use_triton_kernel=True).view(torch.float8_e8m0fnu),
        alpha,
    )


@matmul_mxf8_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(x.shape[0], w.shape[0], dtype=torch.bfloat16)


@torch.library.custom_op("fp_quant::matmul_mxf8_bf16_nn_op", mutates_args=())
def matmul_mxf8_bf16_nn_op(
    x: torch.Tensor,
    w: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return matmul_mxf8_bf16_nn(
        x,
        w,
        to_blocked_qutlass(xs, use_triton_kernel=True),
        to_blocked_qutlass(ws, use_triton_kernel=True).view(torch.float8_e8m0fnu),
        alpha,
    )


@matmul_mxf8_bf16_nn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(x.shape[1], w.shape[0], dtype=torch.bfloat16)


### Backward Quantization


@torch.library.custom_op("fp_quant::backward_t_bf16", mutates_args=())
def backward_t_bf16_op(
    x: torch.Tensor,
    hadamard_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return backward_t_bf16(
        x,
        hadamard_matrix,
    )


@backward_t_bf16_op.register_fake
def _(x, hadamard_matrix):
    xh_e2m1 = torch.empty(
        *x.shape[:-2],
        x.size(-1),
        x.size(-2) // 2,
        dtype=torch.float4_e2m1fn_x2,
        device=x.device,
    )
    xh_e8m0 = torch.empty(
        *x.shape[:-2],
        x.size(-1),
        x.size(-2) // 32,
        dtype=torch.float8_e8m0fnu,
        device=x.device,
    )
    return xh_e2m1, xh_e8m0


@torch.library.custom_op("fp_quant::backward_qt_bf16", mutates_args=())
def backward_qt_bf16_op(
    x_e2m1: torch.Tensor,
    x_e8m0: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    alpha: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return backward_qt_bf16(x_e2m1, x_e8m0, hadamard_matrix, alpha)


@backward_qt_bf16_op.register_fake
def _(x_e2m1, x_e8m0, hadamard_matrix, alpha):
    xh_e2m1 = torch.empty(
        *x_e2m1.shape[:-2],
        x_e2m1.size(-1) * 2,
        x_e2m1.size(-2) // 2,
        dtype=torch.float4_e2m1fn_x2,
        device=x_e2m1.device,
    )
    xh_e8m0 = torch.empty(
        *x_e8m0.shape[:-2],
        x_e8m0.size(-1) * 32,
        x_e8m0.size(-2) // 32,
        dtype=torch.float8_e8m0fnu,
        device=x_e2m1.device,
    )
    return xh_e2m1, xh_e8m0


@torch.library.custom_op("fp_quant::backward_bf16_square_double_mxfp8", mutates_args=())
def backward_bf16_square_double_mxfp8_op(
    x_bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return backward_bf16_square_double_mxfp8(x_bf16)


@backward_bf16_square_double_mxfp8_op.register_fake
def _(x_bf16):
    m, n = x_bf16.shape
    m_up128 = ((m - 1) // 128) * 128 + 128

    x_fp8 = torch.empty(m_up128, n, device=x_bf16.device, dtype=torch.float8_e4m3fn)
    row_scales = torch.empty(
        m_up128,
        n // 32,
        device=x_bf16.device,
        dtype=torch.float8_e8m0fnu,
    )
    column_scales = torch.empty(
        n,
        m_up128 // 32,
        device=x_bf16.device,
        dtype=torch.float8_e8m0fnu,
    )
    return x_fp8, row_scales, column_scales


@torch.library.custom_op("fp_quant::mxfp4_transpose_mxfp8", mutates_args=())
def mxfp4_transpose_mxfp8_op(
    x_fp4: torch.Tensor,
    scales: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return mxfp4_transpose_mxfp8(x_fp4, scales)


@mxfp4_transpose_mxfp8_op.register_fake
def _(x_fp4, scales):
    m, n = x_fp4.size(0), x_fp4.size(1) * 2
    m_up128 = ((m - 1) // 128) * 128 + 128

    x_fp8 = torch.empty(
        n,
        m_up128,
        device=x_fp4.device,
        dtype=torch.float8_e4m3fn,
    )
    shared_exps = torch.empty(
        n,
        m_up128 // 32,
        device=x_fp4.device,
        dtype=torch.float8_e8m0fnu,
    )

    return x_fp8, shared_exps


def to_blocked(x: torch.Tensor) -> torch.Tensor:
    return to_blocked_qutlass(x, use_triton_kernel=True)
