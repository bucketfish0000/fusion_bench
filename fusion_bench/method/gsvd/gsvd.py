import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Optional, cast

import lightning as L
import torch
from omegaconf import DictConfig
from torch import nn

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config
from fusion_bench.models.utils import is_leaf_module
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin
from fusion_bench.taskpool import BaseTaskPool
from fusion_bench.utils import seed_everything_by_time
from fusion_bench.utils.json import save_to_json

from . import utils as gsvd_utils


@auto_register_config
class GSVDMerging(
    BaseAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):
    def __init__(
        self,
        seed: Optional[int] = None,
        shuffle_order: bool = True,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        fairness: float = 0.5,
        prune_policy: str = "SVD",
        preserve_portion: float = 0.8,
        non_matrix_policy: str = "pretrain",
        **kwargs,
    ):
        self.seed = seed
        self.shuffle_order = shuffle_order
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        self.fairness = fairness
        self.prune_policy = prune_policy
        self.preserve_portion = preserve_portion
        self.non_matrix_policy = non_matrix_policy
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        self.modelpool = modelpool
        if self.seed is not None:
            L.seed_everything(self.seed)
        else:
            seed_everything_by_time(self.fabric)

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        if self.log_dir is not None:
            save_to_json(model_names, Path(self.log_dir) / "model_names.json")

        if self.evaluate_on_every_step and getattr(self._program, "taskpool", None) is not None:
            self.taskpool = cast(BaseTaskPool, self._program.taskpool)
            if hasattr(self.taskpool, "_test_datasets"):
                self._test_datasets = deepcopy(self.taskpool._test_datasets)

        with self.profile("loading model"):
            pretrained_model = modelpool.load_pretrained_model()

        merged_model = None
        for model_idx, model_name in enumerate(model_names):
            print(
                f"--------- Merging {model_idx + 1}/{len(model_names)}-th with {model_name} ---------"
            )
            if model_idx == 0:
                with self.profile("loading model"):
                    merged_model = modelpool.load_model(model_name)
            else:
                with self.profile("loading model"):
                    model_2 = modelpool.load_model(model_name)
                with self.profile("merging model"):
                    merged_model = self.merge_gsvd(
                        model_1=merged_model,
                        pretrained_model=pretrained_model,
                        model_2=model_2,
                        first_step=(model_idx == 1),
                    )

            if self.save_on_every_step:
                with self.profile("saving model"):
                    self.save_merged_model(merged_model, model_idx)

            if (
                self.evaluate_on_every_step
                and hasattr(self, "taskpool")
                and hasattr(self, "_test_datasets")
                and hasattr(self.taskpool, "_is_setup")
                and hasattr(self.taskpool, "_test_datasets")
            ):
                with self.profile("evaluating model"):
                    self.taskpool._is_setup = False
                    self.taskpool._test_datasets = DictConfig(
                        {n: self._test_datasets[n] for n in model_names[: model_idx + 1]}
                    )
                    report = self.taskpool.evaluate(deepcopy(merged_model))
                    save_to_json(report, Path(self.log_dir) / f"report_{model_idx}.json")

        self.print_profile_summary()
        return merged_model
    
    def merge_gsvd(
        self,
        model_1: nn.Module,
        pretrained_model: nn.Module,
        model_2: nn.Module,
        first_step: bool,
    ) -> nn.Module:
        """Merge model_1 and model_2 layer by layer into a merged model."""
        covered_parameters = set()
        time_cost = []
        merged_model = model_1

        # Traverse leaf modules so every parameter is handled exactly once.
        for module_name, model_1_module in merged_model.named_modules():
            if not is_leaf_module(model_1_module):
                continue

            pretrained_module = pretrained_model.get_submodule(module_name)
            model_2_module = model_2.get_submodule(module_name)

            # Handle every parameter in the current leaf module.
            for param_name, model_1_param in model_1_module.named_parameters(recurse=False):
                full_param_name = (
                    f"{module_name}.{param_name}" if module_name else param_name
                )

                if isinstance(model_1_module, nn.Linear) and param_name == "weight":
                    import time

                    start_time = time.time()
                    merged_param_value = self._merge_linear_layer(
                        model_1_layer=model_1_module,
                        pretrained_layer=pretrained_module,
                        model_2_layer=model_2_module,
                        first_step=first_step,
                    )
                    end_time = time.time()
                    time_cost.append(end_time - start_time)
                else:
                    merged_param_value = self._merge_non_matrix_parameter(
                        model_1_param=model_1_param,
                        pretrained_param=getattr(pretrained_module, param_name, None),
                        model_2_param=getattr(model_2_module, param_name, None),
                    )

                if merged_param_value is not None:
                    model_1_param.data = merged_param_value.to(
                        device=model_1_param.device,
                        dtype=model_1_param.dtype,
                    )
                covered_parameters.add(full_param_name)

        all_parameters = {name for name, _ in merged_model.named_parameters()}
        missing_parameters = sorted(all_parameters - covered_parameters)
        if missing_parameters:
            raise RuntimeError(
                "merge_gsvd did not cover all model parameters. Missing: "
                + ", ".join(missing_parameters)
            )

        return merged_model

    def _merge_linear_layer(
        self,
        model_1_layer: nn.Linear,
        pretrained_layer: nn.Linear,
        model_2_layer: nn.Linear,
        first_step: bool,
    ) -> torch.Tensor:
        """Merge one linear layer with its matching counterparts."""
        eps = 1e-6
        model_1_weight = model_1_layer.weight.detach()
        pretrained_weight = pretrained_layer.weight.detach()
        model_2_weight = model_2_layer.weight.detach()

        delta_W_1 = model_1_weight - pretrained_weight
        delta_W_2 = model_2_weight - pretrained_weight

        if torch.norm(delta_W_1) < eps or torch.norm(delta_W_2) < eps:
            merged_weight = self._merge_non_matrix_parameter(
                model_1_param=model_1_weight,
                pretrained_param=pretrained_weight,
                model_2_param=model_2_weight,
            )
            if merged_weight is None:
                raise RuntimeError("Linear layer weight unexpectedly merged to None.")
            return merged_weight

        delta_W_1_pruned, rank_1 = gsvd_utils.prune_matrix(
            delta_W_1,
            prune_policy=self.prune_policy,
            preserve_portion=self.preserve_portion,
            first_step=first_step,
        )
        delta_W_2_pruned, rank_2 = gsvd_utils.prune_matrix(
            delta_W_2,
            prune_policy=self.prune_policy,
            preserve_portion=self.preserve_portion,
            first_step=True,
        )

        U_1, Sigma_1, U_2, Sigma_2, M, R, Q = gsvd_utils.gsvd(
            delta_W_1_pruned,
            delta_W_2_pruned,
            prune_policy=self.prune_policy,
            preserve_portion=self.preserve_portion,
            rank_1=rank_1,
            rank_2=rank_2,
        )

        Sigma_merged, U_merged = self._resolve_conflict(
            Sigma_1=Sigma_1,
            Sigma_2=Sigma_2,
            U_1=U_1,
            U_2=U_2,
        )
        Vh = self._construct_right_factor(M=M, R=R, Q=Q)
        merged_delta_W = U_merged @ Sigma_merged @ Vh
        return pretrained_weight + merged_delta_W

    def _construct_optimized_left_basis(
        self,
        m: int,
        U_1: torch.Tensor,
        U_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        L = U_1.T @ U_2
        C, cos_theta, Sh = torch.linalg.svd(L)
        k = cos_theta.numel()

        C = C[:, :k]
        S = Sh.T[:, :k]
        A = U_1 @ C
        B = U_2 @ S

        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta.square(), min=0.0))
        delta_seq = 0.5 * torch.pi - torch.arccos(cos_theta)

        Omega_1 = torch.eye(m, dtype=U_1.dtype, device=U_1.device)
        Omega_2 = torch.eye(m, dtype=U_2.dtype, device=U_2.device)

        eps = 1e-5
        for i in range(k):
            cos_i = cos_theta[i]
            if cos_i.square() < eps or (1.0 - cos_i.square()) < eps:
                continue

            sin_i = sin_theta[i]
            a_i = A[:, i]
            b_i = B[:, i]
            e1 = a_i / a_i.norm()
            e2 = (b_i / b_i.norm() - cos_i * e1) / (sin_i + eps)

            delta = delta_seq[i]
            Omega_1 = gsvd_utils.rotation_on_plane(m, e1, e2, -delta) @ Omega_1
            Omega_2 = gsvd_utils.rotation_on_plane(m, e2, e1, -delta) @ Omega_2

        U_1_prime = Omega_1 @ U_1
        U_2_prime = Omega_2 @ U_2
        return U_1_prime, U_2_prime

    def _resolve_conflict(
        self,
        Sigma_1: torch.Tensor,
        Sigma_2: torch.Tensor,
        U_1: torch.Tensor,
        U_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Sigma_1_hat, U_1_hat = gsvd_utils.row_col_select(Sigma_1, U_1)
        Sigma_2_hat, U_2_hat = gsvd_utils.row_col_select(Sigma_2, U_2)

        if Sigma_1_hat.numel() == 0:
            return Sigma_2_hat, U_2_hat
        if Sigma_2_hat.numel() == 0:
            return Sigma_1_hat, U_1_hat

        U_1_orth, U_2_orth = self._construct_optimized_left_basis(
            U_1.shape[0], U_1_hat, U_2_hat
        )

        Sigma_1_prime = U_1_orth.T @ U_1_hat @ Sigma_1_hat
        Sigma_2_prime = U_2_orth.T @ U_2_hat @ Sigma_2_hat
        Sigma_merged = torch.cat([Sigma_1_prime, Sigma_2_prime], dim=0)
        U_merged_left = U_1_orth
        U_merged_right = U_2_orth
        U_merged = torch.cat([U_merged_left, U_merged_right], dim=1)
        return Sigma_merged, U_merged

    def _construct_right_factor(
        self,
        M: torch.Tensor,
        R: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        MhR = M.T @ R
        n = Q.shape[0]
        k = MhR.shape[0]
        if k < n:
            pad = torch.zeros((k, n - k), device=MhR.device, dtype=MhR.dtype)
            MhR = torch.cat([MhR, pad], dim=1)
        Vh = MhR @ Q.T
        return Vh


    def _merge_non_matrix_parameter(
        self,
        model_1_param: Optional[torch.Tensor],
        pretrained_param: Optional[torch.Tensor],
        model_2_param: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Merge a non-matrix parameter such as a bias vector."""
        if model_1_param is None:
            return None
        if pretrained_param is None or model_2_param is None:
            return model_1_param

        delta_W_1 = model_1_param.detach() - pretrained_param.detach()
        delta_W_2 = model_2_param.detach() - pretrained_param.detach()

        if self.non_matrix_policy == "average":
            merged_delta_W = (
                self.fairness * delta_W_1 + (1.0 - self.fairness) * delta_W_2
            )
        elif self.non_matrix_policy == "pretrain":
            merged_delta_W = torch.zeros_like(pretrained_param.detach())
        else:
            raise ValueError(
                "Unsupported non_matrix_policy. Use 'pretrain' or 'average'."
            )

        return pretrained_param.detach() + merged_delta_W

    def save_merged_model(
        self,
        merged_model: nn.Module,
        step: int,
    ):
        if self.log_dir is None:
            return
        save_path = Path(self.log_dir) / "checkpoints" / f"merged_model_{step}"
        os.makedirs(save_path.parent, exist_ok=True)
        self.modelpool.save_model(merged_model, str(save_path))
