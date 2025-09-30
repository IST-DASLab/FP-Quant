from random import randint

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32 * 32}),
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def nvfp4_forward_kernel(
    x_ptr,
    hadamard_matrix_ptr,
    global_scale_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    hadamard_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets_hadamard = tl.arange(0, hadamard_dim * hadamard_dim)
    hadamard_matrix = tl.load(hadamard_matrix_ptr + offsets_hadamard).reshape(
        hadamard_dim, hadamard_dim
    )

    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)

    # hadamard transform
    x = tl.reshape(x_flat, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_had = tl.dot(x, hadamard_matrix)

    # group
    x_had_grouped = tl.reshape(x_had, (BLOCK_SIZE // 16, 16))

    # scale
    if global_scale_ptr is not None:
        global_scale = tl.load(global_scale_ptr)
    else:
        global_scale = 1.0
    vec_max = tl.max(tl.abs(x_had_grouped), axis=-1, keep_dims=True)
    scale = global_scale * vec_max / 6
    scale = scale.to(tl.float8e4nv)
    x_had_scaled = x_had_grouped * global_scale / scale

    # quantize
    x_had_scaled_abs = tl.abs(x_had_scaled)
    x_had_scaled_sign = tl.where(
        x_had_scaled > 0,
        1,
        -1,
    )

    x_fp4 = (
        tl.where(
            x_had_scaled_abs > 5,
            6,
            tl.where(
                x_had_scaled_abs > 3.5,
                4,
                tl.where(
                    x_had_scaled_abs > 2.5,
                    3,
                    tl.where(
                        x_had_scaled_abs > 1.75,
                        2,
                        tl.where(
                            x_had_scaled_abs > 1.25,
                            1.5,
                            tl.where(
                                x_had_scaled_abs > 0.75,
                                1,
                                tl.where(
                                    x_had_scaled_abs > 0.25,
                                    0.5,
                                    0,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        * x_had_scaled_sign
    )

    # dequantize
    x_dequantized = x_fp4 * scale
    x_dequantized /= global_scale

    # Reshape back to flat form for storage
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))

    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def nvfp4_forward_kernel_wrapper(
    x,
    hadamard_matrix,
    global_scale,
):
    # Make sure inputs are contiguous
    x = x.contiguous()

    # Create output tensor
    output = torch.empty_like(x)

    # Get total number of elements and calculate grid for launching the kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch optimized kernel
    with torch.cuda.device(x.device):
        nvfp4_forward_kernel[grid](
            x_ptr=x,
            hadamard_matrix_ptr=hadamard_matrix,
            global_scale_ptr=global_scale,
            output_ptr=output,
            n_elements=n_elements,
            hadamard_dim=hadamard_matrix.shape[-1],
        )

    return output
