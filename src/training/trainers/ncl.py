from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from src.training.trainers.base import BaseEnsembleTrainer
from src.training.trainers.utils import (
    _maybe_squeeze_for_regression,
    params_per_model,
    zeros_like_params_list,
    grads_or_zeros,
    dot_param_lists,
)


class _NCLBase(BaseEnsembleTrainer):
    def _normalize_outputs(
        self, outputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        if (
            self.cfg.task_type == "classification"
            and self.cfg.normalize == "softmax"
        ):
            return [torch.softmax(o, dim=1) for o in outputs]
        if (
            self.cfg.task_type == "classification"
            and self.cfg.normalize == "bn"
        ):
            return [
                (o - o.mean(0, keepdim=True))
                / (o.std(0, keepdim=True, unbiased=False) + 1e-6)
                for o in outputs
            ]
        return outputs

    def _compute_ncl_penalty(
        self, outputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        outs_norm = self._normalize_outputs(outputs)
        mean_norm = torch.stack(outs_norm).mean(0).detach()
        return [(out - mean_norm).pow(2).mean() for out in outs_norm]

    def _compute_dncc_penalty(
        self,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
        eps: float = 1e-8,
    ) -> List[torch.Tensor]:
        """Computes the Deep Negative Correlation Classification (DNCC) diversity penalty.

        This function implements the core idea of Equation (8) from the DNCC paper,
        where a penalty term is applied to decorrelate ensemble members by comparing
        each model's log-probability for the correct class against the ensemble's
        average log-probability for that same class.

        **Note**: this penalty is specifically designed for classification tasks.

        Args:
            outputs: A list of logits tensors with shape (B, K), one per
                ensemble member.
            targets: A tensor of shape (B,) containing the true class labels.

        Returns:
            A list containing one scalar penalty per ensemble member, each measuring
            how similar that model's correct-class log-probability is to the ensemble's
            average correct-class log-probability.
        """
        # probas
        probs = [out.softmax(dim=1) for out in outputs]  # list (B, C)

        B = targets.size(0)
        idx = torch.arange(B, device=targets.device)

        # per-model true-class probas
        p_true_list = [p[idx, targets] for p in probs]  # list (B,)

        # ens mean
        with torch.no_grad():
            mean_p = torch.stack(p_true_list, dim=0).mean(dim=0)  # (B,)
            mean_p = mean_p.clamp_min(eps)

        penalties = []
        for p_true in p_true_list:
            p_true = p_true.clamp_min(eps)

            # d_{-log}(p, q) = -log(p) + log(q) + (p - q) / q
            div = (
                -torch.log(p_true)
                + torch.log(mean_p)
                + (p_true - mean_p) / mean_p
            )
            penalties.append(div.mean())

        return penalties

    def _get_logs(
        self,
        task_losses: List[torch.Tensor],
        penalties: List[torch.Tensor],
        total_losses: List[torch.Tensor],
    ) -> Dict[str, float]:
        task_mean = (
            torch.stack([tl.detach() for tl in task_losses]).mean().item()
        )
        pen_mean = torch.stack([p.detach() for p in penalties]).mean().item()
        loss_mean = (
            torch.stack([L.detach() for L in total_losses]).mean().item()
        )
        return {
            "train_loss": loss_mean,
            "task": task_mean,
            "penalty": pen_mean,
        }


class NCLTrainer(_NCLBase):
    def __init__(
        self,
        models: List[nn.Module],
        lambda_ncl: float = 0.5,
        use_dncc: bool = False,
    ) -> None:
        super().__init__(models)
        self.lambda_ncl = float(lambda_ncl)
        self.use_dncc = use_dncc

    def _step_fn(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        val_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> dict:
        xb, yb = batch
        outputs = self(xb)
        outputs = [
            _maybe_squeeze_for_regression(o, yb, self.cfg.task_type)
            for o in outputs
        ]

        if self.use_dncc and self.cfg.task_type == "classification":
            penalties = self._compute_dncc_penalty(outputs, yb)
        elif self.use_dncc and self.cfg.task_type != "classification":
            raise ValueError(
                "DNCC penalty can only be used for classification."
            )
        else:
            penalties = self._compute_ncl_penalty(outputs)

        task_losses = [self.criterion(o, yb) for o in outputs]
        total_losses = [
            tl - self.lambda_ncl * pen
            for tl, pen in zip(task_losses, penalties)
        ]

        for opt, loss in zip(self.optimizers, total_losses):
            opt.zero_grad()
            loss.backward()
            opt.step()

        return self._get_logs(task_losses, penalties, total_losses)


class AdaptiveNCLTrainer(_NCLBase):
    def __init__(
        self,
        models: Sequence[nn.Module],
        lambda_init: float = 0.5,
        lambda_lr: float = 1e-2,
        lambda_step_every: int = 1,
        project_lambda_nonneg: bool = True,
        use_dncc: bool = False,
        hvp_damping: float = 0.0,  # no damping by default
    ) -> None:
        super().__init__(models)
        self.lambda_ncl = float(lambda_init)
        self.lambda_lr = float(lambda_lr)
        self.lambda_step_every = lambda_step_every
        self.project_lambda_nonneg = project_lambda_nonneg
        self.use_dncc = use_dncc
        self.hvp_damping = hvp_damping
        self._global_step = 0

    def _step_fn(self, batch, val_batch):
        if val_batch is None:
            raise ValueError("val_batch must be provided for this trainer.")

        x, y = batch
        device = x.device

        # Lazy init
        if self._global_step == 0:
            self._per_model_params = params_per_model(self.models)
            self._lambda_param = torch.tensor([self.lambda_ncl], device=device)
            self._Z_list = zeros_like_params_list(self._per_model_params)

        self._global_step += 1

        # Forward
        outs = [m(x) for m in self.models]
        outs = [
            _maybe_squeeze_for_regression(o, y, self.cfg.task_type)
            for o in outs
        ]
        L_list = [self.criterion(o, y) for o in outs]

        if self.use_dncc and self.cfg.task_type == "classification":
            R_list = self._compute_dncc_penalty(outs, y)
        elif self.use_dncc and self.cfg.task_type != "classification":
            raise ValueError(
                "DNCC penalty can only be used for classification."
            )
        else:
            R_list = self._compute_ncl_penalty(outs)
        J_list = [
            L - float(self._lambda_param.item()) * R
            for L, R in zip(L_list, R_list)
        ]

        # Grad of J_i wrt params_i
        grads_list = []
        for params_i, J_i in zip(self._per_model_params, J_list):
            grads_i = torch.autograd.grad(
                J_i, params_i, create_graph=True, allow_unused=True
            )
            grads_list.append(grads_or_zeros(grads_i, params_i))

        # HVP: A_i Z_i = Z_i - H_i Z_i
        Hv_list = []
        for grads_i, Z_i, params_i in zip(
            grads_list, self._Z_list, self._per_model_params
        ):
            dot_gZ_i = dot_param_lists(grads_i, Z_i)
            Hv_i_raw = torch.autograd.grad(
                dot_gZ_i, params_i, retain_graph=True, allow_unused=True
            )
            Hv_i = grads_or_zeros(Hv_i_raw, params_i)
            if self.hvp_damping > 0.0:
                Hv_i = [h + self.hvp_damping * z for h, z in zip(Hv_i, Z_i)]
            Hv_list.append(Hv_i)

        # B_i
        B_list = []
        for R_i, params_i, optim_i in zip(
            R_list, self._per_model_params, self.optimizers
        ):
            gRi = torch.autograd.grad(
                R_i, params_i, retain_graph=True, allow_unused=True
            )
            B_list.append(
                [
                    optim_i.param_groups[0]["lr"] * g
                    for g in grads_or_zeros(gRi, params_i)
                ]
            )

        # Update Z_i + collect norms for logging
        (
            Z_norms,
            B_norms,
            Hv_norms,
            A_dot_Z_norms,
        ) = [], [], [], []
        for i, optim_i in enumerate(self.optimizers):
            A_dot_Z_i = [
                z - optim_i.param_groups[0]["lr"] * h
                for z, h in zip(self._Z_list[i], Hv_list[i])
            ]
            self._Z_list[i] = [a + b for a, b in zip(A_dot_Z_i, B_list[i])]
            # moving-average-damped style
            # self._Z_list[i] = [.95*a + b for a, b in zip(A_dot_Z_i, B_list[i])]

            # Norms
            with torch.no_grad():
                z_norm = torch.sqrt(sum((z**2).sum() for z in self._Z_list[i]))
                b_norm = torch.sqrt(sum((b**2).sum() for b in B_list[i]))
                hv_norm = torch.sqrt(sum((h**2).sum() for h in Hv_list[i]))
                a_dot_z_norm = torch.sqrt(sum((a**2).sum() for a in A_dot_Z_i))
                Z_norms.append(float(z_norm))
                B_norms.append(float(b_norm))
                Hv_norms.append(float(hv_norm))
                A_dot_Z_norms.append(float(a_dot_z_norm))

        # SGD update params_i
        grad_norms = []
        with torch.no_grad():
            for params_i, grads_i, optim_i in zip(
                self._per_model_params, grads_list, self.optimizers
            ):
                for p, g in zip(params_i, grads_i):
                    p.add_(-optim_i.param_groups[0]["lr"] * g)

                if g is not None:
                    grad_norm = torch.sqrt(sum((g**2).sum() for g in grads_i))
                    grad_norms.append(float(grad_norm))

        # Update lambda
        self.eval_mode()
        if self._global_step % max(1, self.lambda_step_every) == 0:
            xv, yv = val_batch
            outs_val = [m(xv) for m in self.models]
            outs_val = [
                _maybe_squeeze_for_regression(o, yv, self.cfg.task_type)
                for o in outs_val
            ]
            fbar = torch.stack(outs_val, 0).mean(0)
            E_t = self.criterion(fbar, yv)

            # Grad of E wrt params_i
            dE_list = []
            for params_i in self._per_model_params:
                dE_i_raw = torch.autograd.grad(
                    E_t, params_i, retain_graph=True, allow_unused=True
                )
                dE_list.append(grads_or_zeros(dE_i_raw, params_i))

            # HGD update
            g_lambda = sum(
                dot_param_lists(dE_i, Z_i)
                for dE_i, Z_i in zip(dE_list, self._Z_list)
            )
            with torch.no_grad():
                self._lambda_param -= self.lambda_lr * g_lambda

                if self.project_lambda_nonneg:
                    self._lambda_param.clamp_(min=0.0)

            # restore train mode for next inner step
            self.train_mode()
        else:
            # still compute g_lambda for logging even if we didn't step lambda
            with torch.no_grad():
                g_lambda = torch.tensor(0.0, device=device)

        # logs
        with torch.no_grad():
            mean_logits = torch.stack(outs, 0).mean(0)
            loss_to_log = self.criterion(mean_logits, y).item()
            lambda_now = float(self._lambda_param.item())

        logs = {
            "loss": loss_to_log,
            "lambda": lambda_now,
            "Z_norm_mean": float(torch.tensor(Z_norms).mean().item()),
            "B_norm_mean": float(torch.tensor(B_norms).mean().item()),
            "Hv_norm_mean": float(torch.tensor(Hv_norms).mean().item()),
            "A_dot_Z_norm_mean": float(
                torch.tensor(A_dot_Z_norms).mean().item()
            ),
            "g_lambda": float(g_lambda.detach().item()),
            "grad_norm_mean": float(torch.tensor(grad_norms).mean().item()),
        }

        return logs


class MultiAdaptiveNCLTrainer(_NCLBase):
    def __init__(
        self,
        models: Sequence[nn.Module],
        lambda_init: float = 0.5,
        lambda_lr: float = 1e-2,
        lambda_step_every: int = 10,
        project_lambda_nonneg: bool = True,
        use_dncc: bool = False,
        hvp_damping: float = 0.0,  # no damping by default
    ) -> None:
        super().__init__(models)
        self.lambda_lr = float(lambda_lr)
        self._lambda_init = float(lambda_init)
        self.lambda_step_every = int(lambda_step_every)
        self.project_lambda_nonneg = bool(project_lambda_nonneg)
        self.use_dncc = use_dncc
        self.hvp_damping = hvp_damping
        self._global_step = 0

    def _step_fn(self, batch, val_batch):
        if val_batch is None:
            raise ValueError("val_batch must be provided for this trainer.")

        x, y = batch
        device = x.device

        # Lazy init
        if self._global_step == 0:
            self._per_model_params = params_per_model(self.models)
            self._lambda_vec = torch.tensor(
                [self._lambda_init for _ in self.models], device=device
            )
            self._Z_list = zeros_like_params_list(self._per_model_params)

        self._global_step += 1

        # Forward
        outs = [m(x) for m in self.models]
        outs = [
            _maybe_squeeze_for_regression(o, y, self.cfg.task_type)
            for o in outs
        ]
        L_list = [self.criterion(o, y) for o in outs]

        if self.use_dncc and self.cfg.task_type == "classification":
            R_list = self._compute_dncc_penalty(outs, y)
        elif self.use_dncc and self.cfg.task_type != "classification":
            raise ValueError(
                "DNCC penalty can only be used for classification."
            )
        else:
            R_list = self._compute_ncl_penalty(outs)

        J_list = [
            L - float(self._lambda_vec[i]) * R
            for i, (L, R) in enumerate(zip(L_list, R_list))
        ]

        # Grad of J_i wrt params_i
        grads_list = []
        for params_i, J_i in zip(self._per_model_params, J_list):
            g_i = torch.autograd.grad(
                J_i, params_i, create_graph=True, allow_unused=True
            )
            grads_list.append(grads_or_zeros(g_i, params_i))

        # HVP
        Hv_list = []
        for grads_i, Z_i, params_i in zip(
            grads_list, self._Z_list, self._per_model_params
        ):
            dot_gZ = dot_param_lists(grads_i, Z_i)
            Hv_raw = torch.autograd.grad(
                dot_gZ, params_i, retain_graph=True, allow_unused=True
            )
            Hv_i = grads_or_zeros(Hv_raw, params_i)
            if self.hvp_damping > 0.0:
                Hv_i = [h + self.hvp_damping * z for h, z in zip(Hv_i, Z_i)]
            Hv_list.append(Hv_i)

        # B_i
        B_list = []
        for R_i, params_i, optim_i in zip(
            R_list, self._per_model_params, self.optimizers
        ):
            gRi_raw = torch.autograd.grad(
                R_i, params_i, retain_graph=True, allow_unused=True
            )
            gRi = grads_or_zeros(gRi_raw, params_i)
            lr_i = optim_i.param_groups[0]["lr"]
            B_list.append([lr_i * g for g in gRi])

        # Update Z_i + collect norms
        (
            Z_norms,
            B_norms,
            Hv_norms,
            A_dot_Z_norms,
            total_norms,
            Z_scaled_norms,
        ) = [], [], [], [], [], []
        for i, (Z_i, Hv_i, optim_i, B_i) in enumerate(
            zip(self._Z_list, Hv_list, self.optimizers, B_list)
        ):
            lr_i = optim_i.param_groups[0]["lr"]
            A_dot_Z_i = [z - lr_i * h for z, h in zip(Z_i, Hv_i)]
            self._Z_list[i] = [a + b for a, b in zip(A_dot_Z_i, B_i)]

            with torch.no_grad():
                z_norm = torch.sqrt(sum((z**2).sum() for z in self._Z_list[i]))
                b_norm = torch.sqrt(sum((b**2).sum() for b in B_i))
                hv_norm = torch.sqrt(sum((h**2).sum() for h in Hv_i))
                a_dot_z_norm = torch.sqrt(sum((a**2).sum() for a in A_dot_Z_i))
                Z_norms.append(float(z_norm))
                B_norms.append(float(b_norm))
                Hv_norms.append(float(hv_norm))
                A_dot_Z_norms.append(float(a_dot_z_norm))

        # Update params_i
        with torch.no_grad():
            for params_i, grads_i, optim_i in zip(
                self._per_model_params, grads_list, self.optimizers
            ):
                lr_i = optim_i.param_groups[0]["lr"]
                for p, g in zip(params_i, grads_i):
                    p.add_(-lr_i * g)

        # Update lambda_i
        self.eval_mode()
        if self._global_step % max(1, self.lambda_step_every) == 0:
            xv, yv = val_batch
            xv, yv = xv.to(device), yv.to(device)

            outs_val = [m(xv) for m in self.models]
            outs_val = [
                _maybe_squeeze_for_regression(o, yv, self.cfg.task_type)
                for o in outs_val
            ]
            fbar_val = torch.stack(outs_val, 0).mean(0)
            E_t = self.criterion(fbar_val, yv)

            dE_list = []
            for params_i in self._per_model_params:
                dE_raw = torch.autograd.grad(
                    E_t, params_i, retain_graph=True, allow_unused=True
                )
                dE_list.append(grads_or_zeros(dE_raw, params_i))

            g_lambda_vec = torch.stack(
                [
                    dot_param_lists(dE_i, Z_i)
                    for dE_i, Z_i in zip(dE_list, self._Z_list)
                ]
            )

            with torch.no_grad():
                self._lambda_vec -= self.lambda_lr * g_lambda_vec

                if self.project_lambda_nonneg:
                    self._lambda_vec.clamp_(min=0.0)

            # restore train mode for next inner step
            self.train_mode()
        else:
            with torch.no_grad():
                g_lambda_vec = torch.zeros(len(self.models), device=device)

        with torch.no_grad():
            mean_logits = torch.stack(outs, 0).mean(0)
            log_loss = self.criterion(mean_logits, y).item()
            lam_now = self._lambda_vec.detach().cpu().tolist()

        logs = {
            "loss": log_loss,
            "lambda": lam_now,
            "lambda_mean": float(self._lambda_vec.mean().item()),
            "Z_norm_mean": float(torch.tensor(Z_norms).mean().item()),
            "B_norm_mean": float(torch.tensor(B_norms).mean().item()),
            "Hv_norm_mean": float(torch.tensor(Hv_norms).mean().item()),
            "A_dot_Z_norm_mean": float(
                torch.tensor(A_dot_Z_norms).mean().item()
            ),
            "Z_total_norm_mean": float(
                torch.tensor(total_norms).mean().item()
            ),
            "Z_scaled_norm_mean": float(
                torch.tensor(Z_scaled_norms).mean().item()
            ),
            "g_lambda": g_lambda_vec.detach().cpu().tolist(),
        }

        return logs
