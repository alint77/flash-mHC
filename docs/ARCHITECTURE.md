# flash-mHC Architecture

## Per-layer flow

Given input streams `x_streams` with shape `(T, n, C)`:

1. K1 (Triton): `fused_rmsnorm_project`
   - computes fused RMSNorm + projection into `[h_pre, h_post, a_res]` logits
2. K2 (PyTorch): coefficient math
   - `h_pre = sigmoid(...)`
   - `h_post = 2 * sigmoid(...)`
   - `a_res = softmax(...)`
   - `H_res = a_res @ perm_mat`
   - `H_merged = H_res - h_post outer h_pre`
3. K3 (Triton): `fused_pre_map`
   - computes `layer_input[t,c] = sum_j h_pre[t,j] * x_streams[t,j,c]`
4. Wrapped user layer `f(layer_input)`
5. K4 (Triton): `fused_post_res`
   - computes `H_merged @ x_streams + h_post * layer_output`

Backward for K4 is a single fused kernel:
- `_fused_post_res_bwd_fused_kernel_n4`

## Torch op stack

The Triton kernels are exposed through `torch.library` ops in `ops.py`:
- `flash_mhc::fused_rmsnorm_project`
- `flash_mhc::fused_pre_map`
- `flash_mhc::fused_post_res`

Each op has:
- FakeTensor registration
- CUDA implementation
- custom autograd registration

## Runtime autotune

`src/flash_mhc/kernels.py` includes autotuned wrappers for all active fused kernels:
- K1 fwd / K1 bwd-dx
- K3 fwd / K3 bwd-dx / K3 bwd-hpre
- K4 fwd / K4 bwd-fused

`src/flash_mhc/ops.py` controls runtime dispatch:
- default: runtime autotune ON (`FLASH_MHC_TRITON_AUTOTUNE=1`)
- fallback: legacy hardcoded launch path (`FLASH_MHC_TRITON_AUTOTUNE=0`)
- one-time status logs (`FLASH_MHC_TRITON_AUTOTUNE_STATUS=1`)

For eval/inference, `MHCLiteBlock` temporarily disables autotune to avoid warmup latency.

## Constraints

- Current kernels are specialized for `n=4`.
- Best performance assumes bf16 on CUDA.
