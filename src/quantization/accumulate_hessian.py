import torch
import triton
from triton import language as tl


def _get_cuda_autotune_config() -> list[triton.Config]:
    return [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 2}, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=2, num_stages=5, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=2, num_stages=5, num_ctas=1, maxnreg=None),
    ]


@triton.autotune(
    configs=_get_cuda_autotune_config(),
    key=['size_hidden', 'size_batch', 'save_lower_only', 'compute_lower_only', 'size_meta_batch'],
    restore_value=['mat_hessian_ptr'],
)
@triton.jit
def accumulate_hessian_triton_kernel(
        mat_hessian_ptr,
        mat_input_ptr,
        size_hidden: int,
        size_batch: int,
        save_lower_only,
        compute_lower_only,
        size_meta_batch: int,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
) -> None:
    a_ptr, b_ptr, c_ptr = mat_input_ptr, mat_input_ptr, mat_hessian_ptr
    M, N, K = size_hidden, size_hidden, size_batch
    stride_am, stride_ak = 1, size_hidden
    stride_bk, stride_bn = size_hidden, 1
    stride_cm, stride_cn = size_hidden, 1

    # Kernel for computing the matmul C = A x B. A has shape (M, K), B has shape (K, N) and C has shape (M, N)

    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_pid_mn = num_pid_m * num_pid_n
    meta_batch_id = pid // num_pid_mn
    # pid_within_batch = pid % num_pid_mn
    group_id = pid % num_pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m

    is_upper = (pid_m + 1) * BLOCK_SIZE_M <= pid_n * BLOCK_SIZE_N
    if compute_lower_only and is_upper:
        return
    is_lower = pid_m * BLOCK_SIZE_M >= (pid_n + 1) * BLOCK_SIZE_N
    is_diag = not (is_lower or is_upper)

    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + meta_batch_id * (size_batch * size_hidden) \
             + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + meta_batch_id * (size_batch * size_hidden) \
             + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # TODO: (unknown reason) tl.load c here makes the kernel 2x slow
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block of fp32 values for higher accuracy.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.)
        # We accumulate along the K dimension.
        c = tl.dot(a, b, c)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + meta_batch_id * (size_hidden * size_hidden) \
             + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c += tl.load(c_ptrs, mask=c_mask)
    tl.store(c_ptrs, c, mask=c_mask)

    if compute_lower_only and not (save_lower_only or is_diag):
        # Warning: BLOCK_SIZE_M:BLOCK_SIZE_N must be 1:n or n:1 for this kind of copying
        ct_ptrs = c_ptr + meta_batch_id * (size_hidden * size_hidden) \
                  + offs_cm[:, None] * stride_cn + offs_cn[None, :] * stride_cm
        tl.store(ct_ptrs, c, mask=c_mask)


def accumulate_hessian(
    mat_hessian: torch.Tensor,
    mat_input: torch.Tensor,
    save_lower_only: bool = False,
    compute_lower_only: bool = True,
) -> torch.Tensor:
    """
    Accumulate the Hessian matrix (fp32) with the outer product of the input tensor (fp16 or bf16)
    mat_hessian: (size_meta_batch, size_hidden, size_hidden), fp32, the Hessian matrix to be accumulated, modified in-place and returned
    mat_input: (size_meta_batch, size_batch, size_hidden), fp16 or bf16, the input tensor
    save_lower_only: bool, whether to save the lower triangle only
    compute_lower_only: bool, whether to compute the lower triangle only (should be set to False only for debugging)
    """

    assert compute_lower_only or not save_lower_only, 'compute_lower_only must be True when save_lower_only is True'
    assert mat_hessian.is_contiguous() and mat_input.is_contiguous()
    *meta_batch_dims, size_batch, size_hidden = mat_input.shape
    size_meta_batch: int = int(torch.as_tensor(meta_batch_dims).prod())
    device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    torch_accelerator_module = getattr(torch, device_type)
    previous_device: torch.device = torch.device(f'{device_type}:{torch_accelerator_module.current_device()}')
    torch_accelerator_module.set_device(mat_input.device)
    grid = lambda meta: (
        size_meta_batch
        * triton.cdiv(size_hidden, meta['BLOCK_SIZE_M'])
        * triton.cdiv(size_hidden, meta['BLOCK_SIZE_N']),
    )
    # Instead of using a 2D grid, flatten the grid to 1D. This avoids the per-dimension limit.
    accumulate_hessian_triton_kernel[grid](
        mat_hessian, mat_input,
        size_hidden, size_batch,
        save_lower_only,
        compute_lower_only,
        size_meta_batch,
    )
    torch_accelerator_module.set_device(previous_device)
    return mat_hessian


