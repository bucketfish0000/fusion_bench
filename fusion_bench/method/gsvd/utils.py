import math
from typing import Tuple

import torch
from torch import Tensor


def prune_low(
    singular_values: Tensor,
    portion: float,
    boost: bool = False,
    inplace: bool = False,
) -> Tuple[Tensor, int]:
    """Keep the smallest prefix that preserves `portion` energy."""
    if singular_values.ndim != 1:
        raise ValueError("`singular_values` must be a 1D tensor.")
    if not (0.0 <= portion <= 1.0):
        raise ValueError("`portion` must be between 0 and 1.")

    values = singular_values if inplace else singular_values.clone()
    energy = values.square()
    total_energy = energy.sum()
    if total_energy <= 0:
        return values.zero_(), 0

    cumulative = torch.cumsum(energy, dim=0)
    k = int(torch.searchsorted(cumulative, portion * total_energy).item())
    k = min(k, values.numel() - 1)

    if k + 1 < values.numel():
        if boost:
            values[k + 1 :] = values[k]
        else:
            values[k + 1 :] = 0

    return values, k + 1


def prune_entries(matrix: Tensor, portion: float) -> Tensor:
    """Keep the largest-magnitude `portion` of entries in `matrix`."""
    if not (0.0 <= portion <= 1.0):
        raise ValueError("`portion` must be between 0 and 1.")

    flat = matrix.abs().reshape(-1)
    n_total = flat.numel()
    n_keep = max(1, math.floor(n_total * portion))

    cutoff, _ = torch.kthvalue(flat, n_total - n_keep + 1)
    return matrix * (matrix.abs() >= cutoff)


def compute_left_basis(
    P2: Tensor,
    beta: Tensor,
    M: Tensor,
    eps: float,
) -> Tensor:
    """Compute the second left basis U_2 for GSVD."""
    p, _ = P2.shape

    mask = beta > eps
    rank = int(mask.sum().item())
    if rank == 0:
        return torch.linalg.qr(torch.randn_like(P2), mode="reduced").Q[:, :p]

    M_r = M[:, -rank:]
    MR = P2 @ M_r
    beta_r = beta[mask]

    U_2_r = MR / beta_r.unsqueeze(0)
    if rank < p:
        Q_full, _ = torch.linalg.qr(U_2_r, mode="complete")
        U_2_0 = Q_full[:, rank:]
        U_2 = torch.cat([U_2_0, U_2_r], dim=1)
    else:
        U_2 = U_2_r

    return U_2[:, -p:]


def gsvd(
    A: Tensor,
    B: Tensor,
    prune_policy: str,
    preserve_portion: float,
    rank_1: int,
    rank_2: int,
) -> Tuple[Tensor, ...]:
    """Generalized SVD for two matrices with optional right-spectrum pruning."""
    eps = 1e-6
    m, _ = A.shape
    p, _ = B.shape

    C = torch.vstack([A, B])
    P_full, r_values, Qh = torch.linalg.svd(C, full_matrices=True)
    Q = Qh.T

    k = int((r_values > eps).sum().item())
    k = min(k, rank_1 + rank_2)

    r_values = r_values[:k]
    P1 = P_full[:m, :k]
    P2 = P_full[m:, :k]

    U_1, alpha, Mh = torch.linalg.svd(P1, full_matrices=True)
    M = Mh.T

    Sigma_1 = torch.zeros((m, k), device=A.device, dtype=A.dtype)
    for j in range(min(m, k)):
        Sigma_1[j, j] = alpha[j]

    beta = torch.sqrt(torch.clamp(1.0 - alpha.square(), min=0.0))
    beta_fill = torch.flip(beta, dims=(0,))
    Sigma_2 = torch.zeros((p, k), device=B.device, dtype=B.dtype)
    for j in range(min(p, k)):
        Sigma_2[j, j] = beta_fill[j]
    Sigma_2 = torch.flip(Sigma_2, dims=(0, 1))

    U_2 = compute_left_basis(P2, beta, M, eps)
    return U_1, Sigma_1, U_2, Sigma_2, M, torch.diag(r_values), Q


def row_col_select(Sigma: Tensor, U: Tensor) -> Tuple[Tensor, Tensor]:
    """Drop near-zero rows in Sigma and matching columns in U."""
    row_max = Sigma.abs().amax(dim=1)
    keep = row_max > 1e-3
    idx = keep.nonzero(as_tuple=True)[0]
    return Sigma[idx, :], U[:, idx]


def rotation_on_plane(m: int, e1: Tensor, e2: Tensor, phi: Tensor) -> Tensor:
    """Rotation matrix in span{e1,e2} embedded in R^m."""
    e1 = e1 / e1.norm()
    e2 = e2 - torch.dot(e1, e2) * e1
    e2 = e2 / e2.norm()

    basis = torch.stack([e1, e2], dim=1)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    rot_minus_i = torch.tensor(
        [[cos_phi - 1.0, -sin_phi], [sin_phi, cos_phi - 1.0]],
        device=e1.device,
        dtype=e1.dtype,
    )
    return (
        torch.eye(m, device=e1.device, dtype=e1.dtype) + basis @ rot_minus_i @ basis.T
    )


def _prune_matrix_by_svd(
    delta_W: Tensor,
    preserve_portion: float,
) -> Tuple[Tensor, int]:
    U, singular_values, Vh = torch.linalg.svd(delta_W, full_matrices=False)
    singular_values_pruned, _ = prune_low(singular_values, preserve_portion)
    delta_W_pruned = U @ torch.diag(singular_values_pruned) @ Vh
    return delta_W_pruned, int(torch.linalg.matrix_rank(delta_W_pruned).item())


def prune_matrix(
    delta_W: Tensor,
    prune_policy: str,
    preserve_portion: float,
    first_step: bool,
) -> Tuple[Tensor, int]:
    if prune_policy == "SVD":
        if first_step:
            return _prune_matrix_by_svd(delta_W, preserve_portion)
        return delta_W, int(torch.linalg.matrix_rank(delta_W).item())
    if prune_policy == "entry":
        delta_W_pruned = prune_entries(delta_W, preserve_portion)
        return delta_W_pruned, int(torch.linalg.matrix_rank(delta_W_pruned).item())
    if prune_policy == "none":
        return delta_W, int(torch.linalg.matrix_rank(delta_W).item())
    raise ValueError(
        "Unsupported prune_policy. Use 'SVD', 'entry', or 'none'."
    )
