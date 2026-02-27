"""Standalone mHC-lite block with optional Triton-fused path.

This module is framework-agnostic: pass any ``layer`` that maps
``(T, C) -> (T, C)`` in its forward method.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from . import ops as _ops  # noqa: F401  # ensure custom ops are registered


def build_permutation_matrices(n: int) -> tuple[torch.Tensor, int]:
    """Return flattened permutation matrices and identity index.

    Output shape is ``(n!, n*n)`` with values in ``{0,1}``.
    """
    perms = list(permutations(range(n)))
    identity_idx = 0
    mats = torch.zeros(len(perms), n * n)
    for i, perm in enumerate(perms):
        for row, col in enumerate(perm):
            mats[i, row * n + col] = 1.0
    return mats, identity_idx


class MHCLiteBlock(nn.Module):
    """mHC-lite wrapper for a residual layer.

    The wrapped ``layer`` should accept ``(T, C)`` and return ``(T, C)``.

    Notes:
    - Triton fused kernels are currently specialized for ``n_streams == 4``.
    - Triton path is used only on CUDA+bfloat16 with shape ``(T, n, C)``.
    """

    def __init__(
        self,
        n_streams: int,
        hidden_size: int,
        layer: nn.Module,
        triton_fused: bool = True,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.n = n_streams
        self.C = hidden_size
        self.nC = n_streams * hidden_size
        self.layer = layer
        self.triton_fused = triton_fused
        self.activation_checkpointing = activation_checkpointing

        n_fact = math.factorial(n_streams)
        self.n_fact = n_fact

        self.alpha_pre = nn.Parameter(torch.tensor([0.01]))
        self.alpha_post = nn.Parameter(torch.tensor([0.01]))
        self.alpha_res = nn.Parameter(torch.tensor([0.01]))

        total_out = n_streams + n_streams + n_fact
        self.W_all = nn.Linear(self.nC, total_out, bias=True)

        perm_flat, self.identity_idx = build_permutation_matrices(n_streams)
        self.register_buffer("perm_mat", perm_flat)

    def _use_activation_checkpointing(self, x_streams: torch.Tensor) -> bool:
        return (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and x_streams.requires_grad
        )

    def _mhc_coeffs_pytorch(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.n
        dt = x_streams.dtype

        x_flat = x_streams.reshape(*x_streams.shape[:-2], self.nC)
        x_norm = F.rms_norm(x_flat, (self.nC,))

        all_proj = F.linear(x_norm, self.W_all.weight.to(dt), None)
        pre_proj, post_proj, res_proj = all_proj.split([n, n, self.n_fact], dim=-1)

        bias = self.W_all.bias.to(dt)
        pre_bias = bias[:n]
        post_bias = bias[n : 2 * n]
        res_bias = bias[2 * n :]

        h_pre = torch.sigmoid(self.alpha_pre.to(dt) * pre_proj + pre_bias)
        h_post = 2.0 * torch.sigmoid(self.alpha_post.to(dt) * post_proj + post_bias)
        a_res = F.softmax(self.alpha_res.to(dt) * res_proj + res_bias, dim=-1)

        H_res = torch.matmul(a_res, self.perm_mat.to(dt)).unflatten(-1, (n, n))
        H_merged = H_res - h_post.unsqueeze(-1) * h_pre.unsqueeze(-2)
        return h_pre, h_post, H_merged

    def _mhc_pre_map_pytorch(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_pre, h_post, H_merged = self._mhc_coeffs_pytorch(x_streams)
        layer_input = torch.matmul(h_pre.unsqueeze(-2), x_streams).squeeze(-2)
        return layer_input, H_merged, h_post

    def _mhc_post_res_pytorch(
        self,
        x_streams: torch.Tensor,
        layer_output: torch.Tensor,
        H_merged: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        return (
            torch.matmul(H_merged, x_streams)
            + h_post.unsqueeze(-1) * layer_output.unsqueeze(-2)
        )

    def _forward_pytorch(self, x_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        use_ckpt = self._use_activation_checkpointing(x_streams)
        if use_ckpt:
            layer_input, H_merged, h_post = _checkpoint(
                lambda x_: self._mhc_pre_map_pytorch(x_),
                x_streams,
                use_reentrant=False,
            )
        else:
            layer_input, H_merged, h_post = self._mhc_pre_map_pytorch(x_streams)

        layer_output = self.layer(layer_input, **kwargs)

        if use_ckpt:
            return _checkpoint(
                lambda x_, lo_, H_, hp_: self._mhc_post_res_pytorch(x_, lo_, H_, hp_),
                x_streams,
                layer_output,
                H_merged,
                h_post,
                use_reentrant=False,
            )
        return self._mhc_post_res_pytorch(x_streams, layer_output, H_merged, h_post)

    def _mhc_coeffs_triton(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.n
        dt = x_streams.dtype
        T = x_streams.shape[0]

        x_flat = x_streams.reshape(T, self.nC)
        all_proj, _inv_rms = torch.ops.flash_mhc.fused_rmsnorm_project(
            x_flat, self.W_all.weight.to(dt)
        )
        pre_proj, post_proj, res_proj = all_proj.split([n, n, self.n_fact], dim=-1)

        bias = self.W_all.bias.to(dt)
        h_pre = torch.sigmoid(self.alpha_pre.to(dt) * pre_proj + bias[:n])
        h_post = 2.0 * torch.sigmoid(self.alpha_post.to(dt) * post_proj + bias[n : 2 * n])
        a_res = F.softmax(self.alpha_res.to(dt) * res_proj + bias[2 * n :], dim=-1)

        H_res = torch.matmul(a_res, self.perm_mat.to(dt)).unflatten(-1, (n, n))
        H_merged = H_res - h_post.unsqueeze(-1) * h_pre.unsqueeze(-2)
        return h_pre, h_post, H_merged

    def _mhc_pre_map_triton(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_pre, h_post, H_merged = self._mhc_coeffs_triton(x_streams)
        layer_input = torch.ops.flash_mhc.fused_pre_map(x_streams, h_pre.float())
        return layer_input, H_merged.float(), h_post.float()

    def _mhc_post_res_triton(
        self,
        x_streams: torch.Tensor,
        layer_output: torch.Tensor,
        H_merged: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.flash_mhc.fused_post_res(
            x_streams, layer_output, H_merged, h_post
        )

    def _forward_triton(self, x_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        use_ckpt = self._use_activation_checkpointing(x_streams)
        autotune_ctx = (
            nullcontext()
            if self.training
            else _ops.disable_autotune_temporarily()
        )

        with autotune_ctx:
            if use_ckpt:
                layer_input, H_merged, h_post = _checkpoint(
                    lambda x_: self._mhc_pre_map_triton(x_),
                    x_streams,
                    use_reentrant=False,
                )
            else:
                layer_input, H_merged, h_post = self._mhc_pre_map_triton(x_streams)

            layer_output = self.layer(layer_input, **kwargs)

            if use_ckpt:
                return _checkpoint(
                    lambda x_, lo_, H_, hp_: self._mhc_post_res_triton(x_, lo_, H_, hp_),
                    x_streams,
                    layer_output,
                    H_merged,
                    h_post,
                    use_reentrant=False,
                )
            return self._mhc_post_res_triton(x_streams, layer_output, H_merged, h_post)

    def forward(self, x_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        use_triton = (
            self.triton_fused
            and x_streams.is_cuda
            and x_streams.dtype == torch.bfloat16
            and x_streams.dim() == 3
            and self.n == 4
        )
        if use_triton:
            return self._forward_triton(x_streams, **kwargs)
        return self._forward_pytorch(x_streams, **kwargs)
