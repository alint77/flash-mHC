"""
mHC-lite Triton kernel microbenchmark.

Benchmarks the individual Triton kernels in:
  src/flash_mhc/kernels.py

This is intended to be much faster than full training+profiling runs and to
support quick iteration when tuning kernel launch configs.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch


def _add_repo_to_path() -> None:
    # Allow running from repo root without installing the package.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))


@dataclass(frozen=True)
class KernelResult:
    name: str
    calls: int
    avg_ms: float
    med_ms: float
    min_ms: float
    max_ms: float
    achieved_gbps: float | None
    min_theoretical_ms: float | None
    efficiency_pct: float | None


def _time_cuda(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _bytes_model(T: int, n: int, C: int, D_out: int) -> dict[str, int]:
    bf16 = 2
    fp32 = 4
    nC = n * C

    def b(elems: int, bp: int) -> int:
        return elems * bp

    # Same mandatory-bytes model used in docs/TUNING_SM120.md.
    return {
        "_fused_rmsnorm_project_fwd_kernel": b(T * nC, bf16)
        + b(D_out * nC, bf16)
        + b(T * D_out, bf16)
        + b(T, fp32),
        "_fused_rmsnorm_project_bwd_dx_kernel": b(T * nC, bf16)
        + b(D_out * nC, bf16)
        + b(T * D_out, bf16)
        + b(T * D_out, bf16)
        + b(T, fp32)
        + b(T * nC, bf16),
        "_fused_pre_map_fwd_kernel": b(T * n * C, bf16) + b(T * n, fp32) + b(T * C, bf16),
        "_fused_pre_map_bwd_dx_kernel": b(T * n, fp32) + b(T * C, bf16) + b(T * n * C, bf16),
        "_fused_pre_map_bwd_hpre_kernel": b(T * n * C, bf16) + b(T * C, bf16) + b(T * n, fp32),
        "_fused_post_res_fwd_kernel_n4": b(T * n * C, bf16)
        + b(T * C, bf16)
        + b(T * n * n, fp32)
        + b(T * n, fp32)
        + b(T * n * C, bf16),
        # Fused xlo+Hhp: grad_out read once (not twice), all other tensors same
        "_fused_post_res_bwd_fused_kernel_n4": b(T * n * C, bf16)  # grad_out (read once)
        + b(T * n * C, bf16)  # x_streams
        + b(T * C, bf16)      # layer_output
        + b(T * n * n, fp32)  # H_merged
        + b(T * n, fp32)      # h_post
        + b(T * n * C, bf16)  # grad_x (write)
        + b(T * C, bf16)      # grad_lo (write)
        + b(T * n * n, fp32)  # grad_H (write)
        + b(T * n, fp32),     # grad_hp (write)
    }


def _summarize(
    name: str,
    times_ms: list[float],
    bytes_mandatory: int | None,
    peak_gbps: float | None,
) -> KernelResult:
    avg = sum(times_ms) / len(times_ms)
    med = statistics.median(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    achieved_gbps = None
    min_theoretical_ms = None
    efficiency_pct = None
    if bytes_mandatory is not None:
        achieved_gbps = (bytes_mandatory / 1e9) / (avg / 1e3)
        if peak_gbps is not None:
            min_theoretical_ms = (bytes_mandatory / 1e9) / peak_gbps * 1e3
            efficiency_pct = min_theoretical_ms / avg * 100.0
    return KernelResult(
        name=name,
        calls=len(times_ms),
        avg_ms=avg,
        med_ms=med,
        min_ms=mn,
        max_ms=mx,
        achieved_gbps=achieved_gbps,
        min_theoretical_ms=min_theoretical_ms,
        efficiency_pct=efficiency_pct,
    )


def main() -> None:
    _add_repo_to_path()
    import triton
    from flash_mhc import kernels as k

    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=65536)
    ap.add_argument("--C", type=int, default=2560)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peak-gbps", type=float, default=1792.0, help="Peak DRAM BW for efficiency calc.")
    ap.add_argument("--no-eff", action="store_true", help="Disable BW/efficiency calculations.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    T, C, n = args.T, args.C, args.n
    if n != 4:
        raise SystemExit("This benchmark currently assumes n=4 (kernel specializations).")
    nC = n * C
    D_out = 2 * n + 24

    # Inputs
    x_streams = torch.randn((T, n, C), device=device, dtype=torch.bfloat16)
    x_flat = x_streams.reshape(T, nC).contiguous()
    W = torch.randn((D_out, nC), device=device, dtype=torch.bfloat16)

    h_pre = torch.randn((T, n), device=device, dtype=torch.float32)
    h_post = torch.randn((T, n), device=device, dtype=torch.float32)
    H = torch.randn((T, n, n), device=device, dtype=torch.float32)
    layer_output = torch.randn((T, C), device=device, dtype=torch.bfloat16)

    # Grads
    grad_out_proj = torch.randn((T, D_out), device=device, dtype=torch.bfloat16)
    grad_out_pre = torch.randn((T, C), device=device, dtype=torch.bfloat16)
    grad_out_post = torch.randn((T, n, C), device=device, dtype=torch.bfloat16).contiguous()

    # Common config
    NUM_SMS, nw_default, ns = k._get_hw_config()
    cc_major, _ = torch.cuda.get_device_capability()

    bytes_model = _bytes_model(T=T, n=n, C=C, D_out=D_out)
    peak_gbps = None if args.no_eff else args.peak_gbps

    results: list[KernelResult] = []

    # ---- K1 fwd: rmsnorm_project ----
    out_proj = torch.empty((T, D_out), device=device, dtype=torch.bfloat16)
    inv_rms = torch.empty((T,), device=device, dtype=torch.float32)
    BLOCK_K = min(128, _next_pow2(nC))
    BLOCK_T = 128
    grid = (triton.cdiv(T, BLOCK_T),)

    def run_k1_fwd():
        k._fused_rmsnorm_project_fwd_kernel[grid](
            x_flat,
            W,
            out_proj,
            inv_rms,
            T,
            nC,
            D_out,
            BLOCK_T=BLOCK_T,
            BLOCK_K=BLOCK_K,
            num_warps=nw_default,
            num_stages=ns,
        )

    times = _time_cuda(run_k1_fwd, iters=args.iters, warmup=args.warmup)
    results.append(_summarize("_fused_rmsnorm_project_fwd_kernel", times, bytes_model["_fused_rmsnorm_project_fwd_kernel"], peak_gbps))

    # ---- K1 bwd_dx ----
    grad_x_flat = torch.empty_like(x_flat)

    BLOCK_T_BWD = 64 if cc_major == 9 else BLOCK_T
    ns_bwd = ns
    grid_bwd = (triton.cdiv(T, BLOCK_T_BWD),)

    def run_k1_bwd_dx():
        k._fused_rmsnorm_project_bwd_dx_kernel[grid_bwd](
            x_flat,
            W,
            grad_out_proj,
            out_proj,
            inv_rms,
            grad_x_flat,
            T,
            nC,
            D_out,
            BLOCK_T=BLOCK_T_BWD,
            BLOCK_K=BLOCK_K,
            num_warps=nw_default,
            num_stages=ns_bwd,
        )

    times = _time_cuda(run_k1_bwd_dx, iters=args.iters, warmup=args.warmup)
    results.append(_summarize("_fused_rmsnorm_project_bwd_dx_kernel", times, bytes_model["_fused_rmsnorm_project_bwd_dx_kernel"], peak_gbps))

    # ---- K3 fwd: pre_map ----
    out_pre = torch.empty((T, C), device=device, dtype=torch.bfloat16)
    BLOCK_T3 = 64
    BLOCK_C3 = 128 if cc_major == 9 else min(256, _next_pow2(C))
    ns_pre = 3 if cc_major == 9 else ns
    grid_sms = (NUM_SMS,)

    def run_k3_fwd():
        k._fused_pre_map_fwd_kernel[grid_sms](
            x_streams,
            h_pre,
            out_pre,
            T,
            C,
            n,
            BLOCK_T=BLOCK_T3,
            BLOCK_C=BLOCK_C3,
            NUM_SMS=NUM_SMS,
            num_warps=nw_default,
            num_stages=ns_pre,
        )

    times = _time_cuda(run_k3_fwd, iters=args.iters, warmup=args.warmup)
    results.append(_summarize("_fused_pre_map_fwd_kernel", times, bytes_model["_fused_pre_map_fwd_kernel"], peak_gbps))

    # ---- K3 bwd_dx ----
    grad_x_streams = torch.empty_like(x_streams)

    # bwd kernels use original BLOCK_C=256 (fewer live tiles, not register-bound)
    BLOCK_C3_bwd = min(256, _next_pow2(C))

    def run_k3_bwd_dx():
        k._fused_pre_map_bwd_dx_kernel[grid_sms](
            h_pre,
            grad_out_pre,
            grad_x_streams,
            T,
            C,
            n,
            BLOCK_T=BLOCK_T3,
            BLOCK_C=BLOCK_C3_bwd,
            NUM_SMS=NUM_SMS,
            num_warps=nw_default,
            num_stages=ns,
        )

    times = _time_cuda(run_k3_bwd_dx, iters=args.iters, warmup=args.warmup)
    results.append(_summarize("_fused_pre_map_bwd_dx_kernel", times, bytes_model["_fused_pre_map_bwd_dx_kernel"], peak_gbps))

    # ---- K3 bwd_hpre ----
    grad_hpre = torch.empty((T, n), device=device, dtype=torch.float32)

    def run_k3_bwd_hpre():
        k._fused_pre_map_bwd_hpre_kernel[grid_sms](
            x_streams,
            grad_out_pre,
            grad_hpre,
            T,
            C,
            n,
            BLOCK_T=BLOCK_T3,
            BLOCK_C=BLOCK_C3_bwd,
            NUM_SMS=NUM_SMS,
            num_warps=nw_default,
            num_stages=ns,
        )

    times = _time_cuda(run_k3_bwd_hpre, iters=args.iters, warmup=args.warmup)
    results.append(_summarize("_fused_pre_map_bwd_hpre_kernel", times, bytes_model["_fused_pre_map_bwd_hpre_kernel"], peak_gbps))

    # ---- K4 fwd: post_res ----
    out_post = torch.empty_like(x_streams)
    BLOCK_T4 = 64 if cc_major >= 9 else 32
    BLOCK_C4 = 128 if C >= 128 else _next_pow2(C)
    nw4 = 8 if cc_major >= 9 else nw_default

    def run_k4_fwd():
        k._fused_post_res_fwd_kernel_n4[grid_sms](
            x_streams,
            layer_output,
            H,
            h_post,
            out_post,
            T,
            C,
            n,
            BLOCK_T=BLOCK_T4,
            BLOCK_C=BLOCK_C4,
            NUM_SMS=NUM_SMS,
            num_warps=nw4,
            num_stages=ns,
        )

    times = _time_cuda(run_k4_fwd, iters=args.iters, warmup=args.warmup)
    results.append(_summarize("_fused_post_res_fwd_kernel_n4", times, bytes_model["_fused_post_res_fwd_kernel_n4"], peak_gbps))

    # ---- K4 bwd_fused (xlo+Hhp combined) ----
    grad_x_fused = torch.empty_like(x_streams)
    grad_lo_fused = torch.empty((T, C), device=device, dtype=torch.bfloat16)
    grad_H_fused = torch.empty((T, n, n), device=device, dtype=torch.float32)
    grad_hp_fused = torch.empty((T, n), device=device, dtype=torch.float32)

    if cc_major == 9:
        BLOCK_TF = 32
        BLOCK_CF = 128
        ns_fused = 2
    else:
        BLOCK_TF = 32
        BLOCK_CF = min(256, _next_pow2(C))
        ns_fused = ns

    def run_k4_bwd_fused():
        k._fused_post_res_bwd_fused_kernel_n4[grid_sms](
            x_streams,
            layer_output,
            H,
            h_post,
            grad_out_post,
            grad_x_fused,
            grad_lo_fused,
            grad_H_fused,
            grad_hp_fused,
            T,
            C,
            n,
            BLOCK_T=BLOCK_TF,
            BLOCK_C=BLOCK_CF,
            NUM_SMS=NUM_SMS,
            num_warps=nw_default,
            num_stages=ns_fused,
        )

    times = _time_cuda(run_k4_bwd_fused, iters=args.iters, warmup=args.warmup)
    results.append(_summarize("_fused_post_res_bwd_fused_kernel_n4", times, bytes_model["_fused_post_res_bwd_fused_kernel_n4"], peak_gbps))

    # ---- Print ----
    print(f"Device: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability()} NUM_SMS={NUM_SMS}")
    print(f"Shape: T={T} n={n} C={C} D_out={D_out} iters={args.iters} warmup={args.warmup}")
    if peak_gbps is not None:
        print(f"Peak BW assumption: {peak_gbps:.1f} GB/s")
    print()
    header = [
        "kernel",
        "avg_ms",
        "med_ms",
        "min_ms",
        "max_ms",
    ]
    if peak_gbps is None:
        header += ["achieved_GBps"]
    else:
        header += ["min_ms@peak", "eff_%", "achieved_GBps"]
    print(",".join(header))
    for r in results:
        row = [r.name, f"{r.avg_ms:.6f}", f"{r.med_ms:.6f}", f"{r.min_ms:.6f}", f"{r.max_ms:.6f}"]
        if peak_gbps is None:
            row.append("" if r.achieved_gbps is None else f"{r.achieved_gbps:.1f}")
        else:
            row.append("" if r.min_theoretical_ms is None else f"{r.min_theoretical_ms:.6f}")
            row.append("" if r.efficiency_pct is None else f"{r.efficiency_pct:.2f}")
            row.append("" if r.achieved_gbps is None else f"{r.achieved_gbps:.1f}")
        print(",".join(row))

    # Tiny delay so runs launched from job scripts don't interleave prints.
    time.sleep(0.05)


if __name__ == "__main__":
    main()
