import torch
import triton
import triton.language as tl
import torch.utils.benchmark as benchmark
from amd_gemm_kernel_prefill import matmul
from amd_gemm_kernel_decode import matmul as matmul_decode

def construct_amd_gemm(M: int, K: int, N: int):
    SCALE_BLOCK_SIZE = 128

    # Per-token on Activations
    a = torch.randn((M, K), dtype=torch.float32, device='cuda')
    a = a.view(M, -1, SCALE_BLOCK_SIZE)
    max_val = a.abs().amax(dim=2).view(M, -1).clamp(1e-4)
    a_scale = max_val.unsqueeze(2) / torch.finfo(torch.float8_e4m3fn).max
    a = (a / a_scale).view(M, K)
    a_scale = a_scale.view(M, -1).T.contiguous().T  # Layout-preserving
    a_fp8 = a.to(torch.float8_e4m3fn)
    a_fp32 = a_fp8.to(torch.float32)

    # Per-block on Weights
    b = torch.randn((K, N), dtype=torch.float32, device='cuda')
    padded_K = triton.cdiv(K, SCALE_BLOCK_SIZE) * SCALE_BLOCK_SIZE
    padded_N = triton.cdiv(N, SCALE_BLOCK_SIZE) * SCALE_BLOCK_SIZE

    b_padded = torch.zeros((padded_K, padded_N), dtype=b.dtype, device=b.device)
    b_padded[:K, :N] = b

    b_view = b_padded.view(-1, SCALE_BLOCK_SIZE,
                           padded_N // SCALE_BLOCK_SIZE, SCALE_BLOCK_SIZE)  # [K_blk, 128, N_blk, 128]

    b_amax = b_view.abs().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    b_scaled = b_view * (torch.finfo(torch.float8_e4m3fn).max / b_amax)
    b = b_scaled.view_as(b_padded)[:K, :N]
    b_scale = (b_amax / torch.finfo(torch.float8_e4m3fn).max).view(b_view.size(0), b_view.size(2))

    b_fp8 = b.to(torch.float8_e4m3fn)
    b_fp32 = b_fp8.to(torch.float32)

    return a_fp8, b_fp8, a_scale, b_scale

def amd_data_parallel_gemm_fn(a_fp8, b_fp8, c, a_scale, b_scale):
    return matmul(a_fp8, b_fp8, c, a_scale, b_scale)

def amd_splitk_gemm_fn(a_fp8, b_fp8, c, a_scale, b_scale):
    return matmul_decode(a_fp8, b_fp8, c, a_scale, b_scale)


num_threads = torch.get_num_threads()
print(f'Benchmarking on {num_threads} threads')
results = []

for m, n, k in ((64, 4096, 14336), (64, 8192, 28672), (64, 16384, 53248)):

    a_fp8, b_fp8, a_scale, b_scale = construct_amd_gemm(m, n, k)
    c = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

    label = 'FP8 GEMM Kernel Performance'
    sub_label = f'm: {m}, n: {n}, k: {k}'

    results.append(benchmark.Timer(
        stmt='amd_data_parallel_gemm_fn(a, b, c, a_scale, b_scale)',
        setup='from __main__ import amd_data_parallel_gemm_fn',
        globals={'a': a_fp8, 'b' : b_fp8, 'c' : c, 'a_scale' : a_scale, 'b_scale' : b_scale},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='AMD FP8 GEMM Data Parallel').blocked_autorange(min_run_time=1))
    
    results.append(benchmark.Timer(
        stmt='amd_splitk_gemm_fn(a, b, c, a_scale, b_scale)',
        setup='from __main__ import amd_splitk_gemm_fn',
        globals={'a': a_fp8, 'b' : b_fp8, 'c' : c, 'a_scale' : a_scale, 'b_scale' : b_scale},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='AMD FP8 GEMM SplitK').blocked_autorange(min_run_time=1))
    
    
compare = benchmark.Compare(results)
compare.print()
