# https://github.com/ROCm/triton/blob/main_perf/python/perf-kernels/gemm.py

import torch
import triton
import triton.language as tl
import sys
import argparse
import pytest
import re

# TODO: Make this an argument, Benchmarking, testing code and kernel helper need to change for it.
SCALE_BLOCK_SIZE = 128


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
#                 'kpack': 2, 'matrix_instr_nonkdim': 16
#             }, num_warps=4, num_stages=2),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
#                 'kpack': 2, 'matrix_instr_nonkdim': 16
#             }, num_warps=8, num_stages=2),
#         triton.Config(
#             {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},
#             num_warps=8, num_stages=2),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
#                 'kpack': 1, 'matrix_instr_nonkdim': 16
#             }, num_warps=8, num_stages=2),
#         triton.Config(
#             {
#                 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0,
#                 'kpack': 1
#             }, num_warps=8, num_stages=2),
#         triton.Config(
#             {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},
#             num_warps=8, num_stages=2),
#         triton.Config(
#             {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
#             num_warps=8, num_stages=2),
#     ],
#     key=['M', 'N', 'K'],
#     use_cuda_graph=True,
# )
# @triton.heuristics({
#     'EVEN_K':
#     lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0, 'GRID_MN':
#     lambda args: triton.cdiv(args['M'], args['BLOCK_SIZE_M']) * triton.cdiv(args['N'], args['BLOCK_SIZE_N'])
# })
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    a_scale_ptr,
    b_scale_ptr,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K*SPLIT_K)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # Create pointers for first block of A and B input matrices
    offs_k =  (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    k_start = pid_k * BLOCK_SIZE_K
    offs_ks = k_start // GROUP_K
    a_scale_ptrs = (a_scale_ptr + offs_am * stride_ascale_m + offs_ks * stride_ascale_k)

    offs_bsn = offs_bn // GROUP_N
    b_scale_ptrs = b_scale_ptr + offs_bsn * stride_bscale_n + offs_ks * stride_bscale_k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, k_tiles):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        k_remaining = K - kk * (BLOCK_SIZE_K * SPLIT_K)


        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining,  other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        b_scale = tl.load(b_scale_ptrs)
        a_scale = tl.load(a_scale_ptrs)

        accumulator += tl.dot(a, b, input_precision="ieee") * a_scale[:, None] * b_scale[None, :]

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * stride_bk * SPLIT_K

        k_cur = kk * (BLOCK_SIZE_K // GROUP_K)
        k_nxt = (kk + 1) * (BLOCK_SIZE_K // GROUP_K)

        offs_ks = k_nxt - k_cur

        b_scale_ptrs += offs_ks * stride_bscale_k
        a_scale_ptrs += offs_ks * stride_ascale_k 


    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, c, mask=c_mask)


# Wrapper for gemm kernel.
def matmul(a, b, c, a_scale, b_scale, scale_a8_b8=None, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions!!!"
    assert (a.element_size()
            >= b.element_size()), "Mixed dtype GEMMs are only supported when data type of a is bigger than b!!!"
    assert (a.is_floating_point() == b.is_floating_point()
            ), "GEMMs between float and integer type tensors are not supported!!!"
    assert (scale_a8_b8 in [None, 'tensor', 'block']), f"Scaling mode {scale_a8_b8} is not supported!!!"
    M, K = a.shape
    K, N = b.shape

    matrix_instr_nonkdim =  16

    SPLIT_K = 2
    BLOCK_M = 16
    BLOCK_N = 128
    BLOCK_K = 128

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), SPLIT_K)
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scale,
        b_scale,
        a_scale.stride(0) if (a_scale is not None) and a_scale.ndim else 0,
        a_scale.stride(1) if (a_scale is not None) and a_scale.ndim else 0,
        b_scale.stride(0) if (b_scale is not None) and b_scale.ndim else 0,
        b_scale.stride(1) if (b_scale is not None) and b_scale.ndim else 0,
        SPLIT_K=SPLIT_K,
        GROUP_K=SCALE_BLOCK_SIZE,
        GROUP_N=SCALE_BLOCK_SIZE,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
        waves_per_eu=2,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )