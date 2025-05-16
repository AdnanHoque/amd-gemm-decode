import pytest
import torch
from amd_gemm_kernel_prefill import matmul as dp_matmul
from amd_gemm_kernel_decode import matmul as splitk_matmul

def construct_amd_gemm(M: int, K: int, N: int):
    SCALE_BLOCK_SIZE = 128

    # Per-token on Activations
    a = torch.randn((M, K), dtype=torch.float32, device='cuda')
    a = a.view(M, -1, SCALE_BLOCK_SIZE)
    max_val = a.abs().amax(dim=2).view(M, -1).clamp(1e-4)
    a_scale = max_val.unsqueeze(2) / torch.finfo(torch.float8_e4m3fnuz).max
    a = (a / a_scale).view(M, K)
    a_scale = a_scale.view(M, -1).T.contiguous().T
    a_fp8 = a.to(torch.float8_e4m3fnuz)

    # Per-block on Weights
    b = torch.randn((K, N), dtype=torch.float32, device='cuda')
    padded_K = torch.div(K + SCALE_BLOCK_SIZE - 1, SCALE_BLOCK_SIZE, rounding_mode='floor') * SCALE_BLOCK_SIZE
    padded_N = torch.div(N + SCALE_BLOCK_SIZE - 1, SCALE_BLOCK_SIZE, rounding_mode='floor') * SCALE_BLOCK_SIZE

    b_padded = torch.zeros((padded_K, padded_N), dtype=b.dtype, device=b.device)
    b_padded[:K, :N] = b
    b_view = b_padded.view(-1, SCALE_BLOCK_SIZE, padded_N // SCALE_BLOCK_SIZE, SCALE_BLOCK_SIZE)
    b_amax = b_view.abs().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    b_scaled = b_view * (torch.finfo(torch.float8_e4m3fnuz).max / b_amax)
    b = b_scaled.view_as(b_padded)[:K, :N]
    b_scale = (b_amax / torch.finfo(torch.float8_e4m3fnuz).max).view(b_view.size(0), b_view.size(2))

    b_fp8 = b.to(torch.float8_e4m3fnuz)

    return a_fp8, b_fp8, a_scale, b_scale

@pytest.mark.parametrize("M,K,N", [
    (64, 4096, 4096),
    # (64, 8192, 8192),
    # (64, 16384, 53248),
])
def test_fp8_gemm_correctness(M, K, N):
    torch.manual_seed(42)

    a_fp8, b_fp8, a_scale, b_scale = construct_amd_gemm(M, K, N)

    c_dp = torch.empty((M, N), device='cuda', dtype=torch.float16)
    c_splitk = torch.empty((M, N), device='cuda', dtype=torch.float16)

    dp_matmul(a_fp8, b_fp8, c_dp, a_scale, b_scale)
    splitk_matmul(a_fp8, b_fp8, c_splitk, a_scale, b_scale)

    torch.testing.assert_close(c_dp, c_splitk, rtol=0.05, atol=0.1)