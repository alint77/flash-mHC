# SM120 Tuning Summary (RTX 5090)

This document records the tuned launch configs used by the **legacy fallback**
path in `src/flash_mhc/ops.py` when runtime autotune is disabled.

Default runtime behavior now uses Triton autotune wrappers with disk caching.
Use `FLASH_MHC_TRITON_AUTOTUNE=0` to force the legacy hardcoded path below.

Shape tuned: `T=65536, C=1024, n=4`

Workflow:
1. baseline benchmark (`scripts/benchmark_kernels.py`)
2. one-shot grid search (`scripts/gridsearch.py`)
3. rerun benchmark with best configs
4. hardcode selected configs in `src/flash_mhc/ops.py`

## Selected launch configs

- `k1_fwd`: `BLOCK_T=128`, `BLOCK_K=64`, `warps=4`, `stages=3`
- `k1_bwd_dx`: `BLOCK_T=64`, `BLOCK_K=128`, `warps=8`, `stages=3`
- `k3_fwd`: `BLOCK_T=128`, `BLOCK_C=64`, `warps=4`, `stages=2`
- `k3_bwd_dx`: `BLOCK_T=64`, `BLOCK_C=256`, `warps=4`, `stages=4`
- `k3_bwd_hpre`: `BLOCK_T=32`, `BLOCK_C=128`, `warps=8`, `stages=4`
- `k4_fwd`: `BLOCK_T=32`, `BLOCK_C=128`, `warps=8`, `stages=3`
- `k4_bwd_fused`: `BLOCK_T=16`, `BLOCK_C=256`, `warps=8`, `stages=4`

## Observed impact (peak BW assumption 1792 GB/s)

- Total avg kernel time: `7.804 ms -> 4.484 ms` (`42.5%` faster)
- Aggregate efficiency (`sum(lower_bound)/sum(measured)`): `48.2% -> 83.9%`
- Main bottleneck fixed:
  - `_fused_post_res_bwd_fused_kernel_n4`: `4.486 ms -> 1.251 ms` (`72.1%` faster)
