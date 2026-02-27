"""Torch custom ops for mHC-lite Triton kernels.

Goal: make the fused Triton kernels usable under torch.compile without Dynamo
graph breaks by exposing them as dispatcher ops with FakeTensor support and a
registered autograd formula.

The CUDA implementations launch kernels from ``kernels.py``. Fake
implementations only allocate outputs with correct shape/dtype/device.
"""

from __future__ import annotations

import torch
import torch.library


_lib = torch.library.Library("flash_mhc", "DEF")

_lib.define(
    "fused_rmsnorm_project(Tensor x_flat, Tensor W) -> (Tensor out, Tensor inv_rms)"
)
_lib.define(
    "fused_rmsnorm_project_bwd_dx(Tensor grad_out, Tensor x_flat, Tensor W, Tensor proj_out, Tensor inv_rms) -> Tensor grad_x"
)

_lib.define("fused_pre_map(Tensor x_streams, Tensor h_pre) -> Tensor out")
_lib.define(
    "fused_pre_map_backward(Tensor grad_out, Tensor x_streams, Tensor h_pre) -> (Tensor grad_x, Tensor grad_h_pre)"
)

_lib.define(
    "fused_post_res(Tensor x_streams, Tensor layer_output, Tensor H_merged, Tensor h_post) -> Tensor out"
)
_lib.define(
    "fused_post_res_backward(Tensor grad_out, Tensor x_streams, Tensor layer_output, Tensor H_merged, Tensor h_post) -> (Tensor grad_x, Tensor grad_layer_output, Tensor grad_H_merged, Tensor grad_h_post)"
)


@torch.library.register_fake("flash_mhc::fused_rmsnorm_project")
def _fused_rmsnorm_project_fake(x_flat: torch.Tensor, W: torch.Tensor):
    T = x_flat.shape[0]
    D_out = W.shape[0]
    out = torch.empty((T, D_out), device=x_flat.device, dtype=x_flat.dtype)
    inv_rms = torch.empty((T,), device=x_flat.device, dtype=torch.float32)
    return out, inv_rms


@torch.library.register_fake("flash_mhc::fused_rmsnorm_project_bwd_dx")
def _fused_rmsnorm_project_bwd_dx_fake(
    grad_out: torch.Tensor,
    x_flat: torch.Tensor,
    W: torch.Tensor,
    proj_out: torch.Tensor,
    inv_rms: torch.Tensor,
):
    return torch.empty_like(x_flat)


@torch.library.register_fake("flash_mhc::fused_pre_map")
def _fused_pre_map_fake(x_streams: torch.Tensor, h_pre: torch.Tensor):
    T, _, C = x_streams.shape
    return torch.empty((T, C), device=x_streams.device, dtype=x_streams.dtype)


@torch.library.register_fake("flash_mhc::fused_pre_map_backward")
def _fused_pre_map_backward_fake(
    grad_out: torch.Tensor, x_streams: torch.Tensor, h_pre: torch.Tensor
):
    grad_x = torch.empty_like(x_streams)
    grad_h_pre = torch.empty_like(h_pre, dtype=torch.float32)
    return grad_x, grad_h_pre


@torch.library.register_fake("flash_mhc::fused_post_res")
def _fused_post_res_fake(
    x_streams: torch.Tensor,
    layer_output: torch.Tensor,
    H_merged: torch.Tensor,
    h_post: torch.Tensor,
):
    return torch.empty_like(x_streams)


@torch.library.register_fake("flash_mhc::fused_post_res_backward")
def _fused_post_res_backward_fake(
    grad_out: torch.Tensor,
    x_streams: torch.Tensor,
    layer_output: torch.Tensor,
    H_merged: torch.Tensor,
    h_post: torch.Tensor,
):
    grad_x = torch.empty_like(x_streams)
    grad_layer_output = torch.empty_like(layer_output)
    grad_H_merged = torch.empty_like(H_merged, dtype=torch.float32)
    grad_h_post = torch.empty_like(h_post, dtype=torch.float32)
    return grad_x, grad_layer_output, grad_H_merged, grad_h_post


@torch.library.impl(_lib, "fused_rmsnorm_project", "CUDA")
def _fused_rmsnorm_project_cuda(x_flat: torch.Tensor, W: torch.Tensor):
    from . import kernels as k

    T, nC = x_flat.shape
    D_out = W.shape[0]
    out = torch.empty((T, D_out), device=x_flat.device, dtype=x_flat.dtype)
    inv_rms = torch.empty((T,), device=x_flat.device, dtype=torch.float32)

    cc_major, _ = torch.cuda.get_device_capability()
    _, nw, ns = k._get_hw_config()
    if cc_major >= 12:
        # Tuned on RTX 5090 (SM120), T=65536/C=1024/n=4.
        BLOCK_T = 128
        BLOCK_K = 64
        nw = 4
        ns = 3
    else:
        BLOCK_K = min(128, k.triton.next_power_of_2(nC))
        BLOCK_T = 128
    grid = (k.triton.cdiv(T, BLOCK_T),)
    k._fused_rmsnorm_project_fwd_kernel[grid](
        x_flat,
        W,
        out,
        inv_rms,
        T,
        nC,
        D_out,
        BLOCK_T=BLOCK_T,
        BLOCK_K=BLOCK_K,
        num_warps=nw,
        num_stages=ns,
    )
    return out, inv_rms


@torch.library.impl(_lib, "fused_rmsnorm_project_bwd_dx", "CUDA")
def _fused_rmsnorm_project_bwd_dx_cuda(
    grad_out: torch.Tensor,
    x_flat: torch.Tensor,
    W: torch.Tensor,
    proj_out: torch.Tensor,
    inv_rms: torch.Tensor,
):
    from . import kernels as k

    T, nC = x_flat.shape
    D_out = W.shape[0]
    grad_out = grad_out.contiguous()

    _, nw, ns = k._get_hw_config()

    cc_major, _ = torch.cuda.get_device_capability()
    if cc_major >= 12:
        BLOCK_T = 64
        BLOCK_K = 128
        nw = 8
        ns_bwd = 3
    elif cc_major == 9:
        BLOCK_T = 64
        BLOCK_K = min(128, k.triton.next_power_of_2(nC))
        ns_bwd = ns
    else:
        BLOCK_T = 128
        BLOCK_K = min(128, k.triton.next_power_of_2(nC))
        ns_bwd = ns

    grad_x = torch.empty_like(x_flat)
    grid = (k.triton.cdiv(T, BLOCK_T),)
    k._fused_rmsnorm_project_bwd_dx_kernel[grid](
        x_flat,
        W,
        grad_out,
        proj_out,
        inv_rms,
        grad_x,
        T,
        nC,
        D_out,
        BLOCK_T=BLOCK_T,
        BLOCK_K=BLOCK_K,
        num_warps=nw,
        num_stages=ns_bwd,
    )
    return grad_x


@torch.library.impl(_lib, "fused_pre_map", "CUDA")
def _fused_pre_map_cuda(x_streams: torch.Tensor, h_pre: torch.Tensor):
    from . import kernels as k

    T, n, C = x_streams.shape
    if n != 4:
        raise ValueError("flash_mhc::fused_pre_map currently supports n=4 only")
    out = torch.empty((T, C), device=x_streams.device, dtype=x_streams.dtype)
    NUM_SMS, nw, ns = k._get_hw_config()
    cc_major, _ = torch.cuda.get_device_capability()
    if cc_major >= 12:
        # Tuned on RTX 5090 (SM120), T=65536/C=1024/n=4.
        BLOCK_T = 128
        BLOCK_C = 64
        nw = 4
        ns_pre = 2
    elif cc_major == 9:
        # Tuned on H100 for shape T=65536, C=1024, n=4.
        BLOCK_T = 128
        BLOCK_C = 128
        nw = 8
        ns_pre = 4
    else:
        # Original heuristic path for non-SM90 devices.
        BLOCK_T = 64
        BLOCK_C = min(256, k.triton.next_power_of_2(C))
        ns_pre = ns
    grid = (NUM_SMS,)
    k._fused_pre_map_fwd_kernel[grid](
        x_streams,
        h_pre,
        out,
        T,
        C,
        n,
        BLOCK_T=BLOCK_T,
        BLOCK_C=BLOCK_C,
        NUM_SMS=NUM_SMS,
        num_warps=nw,
        num_stages=ns_pre,
    )
    return out


@torch.library.impl(_lib, "fused_pre_map_backward", "CUDA")
def _fused_pre_map_backward_cuda(
    grad_out: torch.Tensor, x_streams: torch.Tensor, h_pre: torch.Tensor
):
    from . import kernels as k

    T, n, C = x_streams.shape
    if n != 4:
        raise ValueError("flash_mhc::fused_pre_map_backward supports n=4 only")
    grad_out = grad_out.contiguous()
    NUM_SMS, nw, ns = k._get_hw_config()
    cc_major, _ = torch.cuda.get_device_capability()
    if cc_major >= 12:
        # Tuned on RTX 5090 (SM120), T=65536/C=1024/n=4.
        BLOCK_T_DX = 64
        BLOCK_C_DX = 256
        nw_dx = 4
        ns_dx = 4
        BLOCK_T_HPRE = 32
        BLOCK_C_HPRE = 128
        nw_hpre = 8
        ns_hpre = 4
    else:
        BLOCK_T_DX = 64
        BLOCK_C_DX = min(256, k.triton.next_power_of_2(C))
        nw_dx = nw
        ns_dx = ns
        BLOCK_T_HPRE = BLOCK_T_DX
        BLOCK_C_HPRE = BLOCK_C_DX
        nw_hpre = nw
        ns_hpre = ns
    grid = (NUM_SMS,)

    grad_x = torch.empty_like(x_streams)
    k._fused_pre_map_bwd_dx_kernel[grid](
        h_pre,
        grad_out,
        grad_x,
        T,
        C,
        n,
        BLOCK_T=BLOCK_T_DX,
        BLOCK_C=BLOCK_C_DX,
        NUM_SMS=NUM_SMS,
        num_warps=nw_dx,
        num_stages=ns_dx,
    )

    grad_h_pre = torch.empty((T, n), device=x_streams.device, dtype=torch.float32)
    k._fused_pre_map_bwd_hpre_kernel[grid](
        x_streams,
        grad_out,
        grad_h_pre,
        T,
        C,
        n,
        BLOCK_T=BLOCK_T_HPRE,
        BLOCK_C=BLOCK_C_HPRE,
        NUM_SMS=NUM_SMS,
        num_warps=nw_hpre,
        num_stages=ns_hpre,
    )
    return grad_x, grad_h_pre


@torch.library.impl(_lib, "fused_post_res", "CUDA")
def _fused_post_res_cuda(
    x_streams: torch.Tensor,
    layer_output: torch.Tensor,
    H_merged: torch.Tensor,
    h_post: torch.Tensor,
):
    from . import kernels as k

    T, n, C = x_streams.shape
    if n != 4:
        raise ValueError("flash_mhc::fused_post_res currently supports n=4 only")
    out = torch.empty_like(x_streams)
    NUM_SMS, nw_default, ns = k._get_hw_config()
    cc_major, _ = torch.cuda.get_device_capability()
    if cc_major >= 12:
        # Tuned on RTX 5090 (SM120), T=65536/C=1024/n=4.
        BLOCK_T = 32
        BLOCK_C = 128
        nw = 8
        ns = 3
    else:
        BLOCK_T = 64 if cc_major >= 9 else 32
        BLOCK_C = 128 if C >= 128 else k.triton.next_power_of_2(C)
        nw = 8 if cc_major >= 9 else nw_default
    grid = (NUM_SMS,)
    k._fused_post_res_fwd_kernel_n4[grid](
        x_streams,
        layer_output,
        H_merged,
        h_post,
        out,
        T,
        C,
        n,
        BLOCK_T=BLOCK_T,
        BLOCK_C=BLOCK_C,
        NUM_SMS=NUM_SMS,
        num_warps=nw,
        num_stages=ns,
    )
    return out


@torch.library.impl(_lib, "fused_post_res_backward", "CUDA")
def _fused_post_res_backward_cuda(
    grad_out: torch.Tensor,
    x_streams: torch.Tensor,
    layer_output: torch.Tensor,
    H_merged: torch.Tensor,
    h_post: torch.Tensor,
):
    from . import kernels as k

    T, n, C = x_streams.shape
    if n != 4:
        raise ValueError("flash_mhc::fused_post_res_backward supports n=4 only")
    grad_out = grad_out.contiguous()
    NUM_SMS, nw_default, ns = k._get_hw_config()
    cc_major, _ = torch.cuda.get_device_capability()
    grid = (NUM_SMS,)

    grad_x = torch.empty_like(x_streams)
    grad_layer_output = torch.empty((T, C), device=x_streams.device, dtype=x_streams.dtype)
    grad_H = torch.empty((T, n, n), device=x_streams.device, dtype=torch.float32)
    grad_hp = torch.empty((T, n), device=x_streams.device, dtype=torch.float32)

    if cc_major >= 12:
        # Tuned on RTX 5090 (SM120), T=65536/C=1024/n=4.
        BLOCK_T_F = 16
        BLOCK_C_F = 256
        nw_fused = 8
        ns_fused = 4
    elif cc_major == 9:
        BLOCK_T_F = 32
        BLOCK_C_F = 128
        nw_fused = nw_default
        ns_fused = 2
    else:
        BLOCK_T_F = 32
        BLOCK_C_F = min(256, k.triton.next_power_of_2(C))
        nw_fused = nw_default
        ns_fused = ns
    k._fused_post_res_bwd_fused_kernel_n4[grid](
        x_streams,
        layer_output,
        H_merged,
        h_post,
        grad_out,
        grad_x,
        grad_layer_output,
        grad_H,
        grad_hp,
        T,
        C,
        n,
        BLOCK_T=BLOCK_T_F,
        BLOCK_C=BLOCK_C_F,
        NUM_SMS=NUM_SMS,
        num_warps=nw_fused,
        num_stages=ns_fused,
    )

    return grad_x, grad_layer_output, grad_H, grad_hp


def _setup_save_inputs_outputs(ctx, inputs, output):
    # output can be Tensor or tuple(Tensor,...)
    if isinstance(output, tuple):
        ctx.save_for_backward(*inputs, *output)
    else:
        ctx.save_for_backward(*inputs, output)


def _fused_rmsnorm_project_backward(ctx, grad_out, grad_inv_rms):
    x_flat, W, proj_out, inv_rms = ctx.saved_tensors
    grad_x = torch.ops.flash_mhc.fused_rmsnorm_project_bwd_dx(
        grad_out, x_flat, W, proj_out, inv_rms
    )
    grad_out_scaled = (grad_out * inv_rms[:, None]).to(x_flat.dtype)
    grad_W = torch.matmul(grad_out_scaled.transpose(0, 1), x_flat)
    return grad_x, grad_W


def _fused_pre_map_backward(ctx, grad_out):
    x_streams, h_pre, _out = ctx.saved_tensors
    return torch.ops.flash_mhc.fused_pre_map_backward(grad_out, x_streams, h_pre)


def _fused_post_res_backward(ctx, grad_out):
    x_streams, layer_output, H_merged, h_post, _out = ctx.saved_tensors
    return torch.ops.flash_mhc.fused_post_res_backward(
        grad_out, x_streams, layer_output, H_merged, h_post
    )


torch.library.register_autograd(
    "flash_mhc::fused_rmsnorm_project",
    _fused_rmsnorm_project_backward,
    setup_context=_setup_save_inputs_outputs,
)
torch.library.register_autograd(
    "flash_mhc::fused_pre_map",
    _fused_pre_map_backward,
    setup_context=_setup_save_inputs_outputs,
)
torch.library.register_autograd(
    "flash_mhc::fused_post_res",
    _fused_post_res_backward,
    setup_context=_setup_save_inputs_outputs,
)
