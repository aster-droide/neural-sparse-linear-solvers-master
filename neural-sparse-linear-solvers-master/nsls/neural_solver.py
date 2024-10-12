from typing import Any, Type, Optional, Tuple, List, Dict

import torch
import torch_scatter
import pytorch_lightning as pl
from torch import Tensor

from .metrics import L1Distance, L2Distance, L2Ratio, VectorAngle
from .losses import CosineDistanceLoss


class NeuralSolver(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        optimizer: Type[torch.optim.Optimizer],
        lr_scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
        **scheduler_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scheduler_kwargs = scheduler_kwargs

        self.model = None
        self.criterion = CosineDistanceLoss()
        self.elementwise_metric = torch.nn.L1Loss()
        self.systemwise_metrics = torch.nn.ModuleDict(
            {
                "l2_ratio": L2Ratio(),
                "l2_distance": L2Distance(),
                "l1_distance": L1Distance(),
                "angle": VectorAngle(),
            }
        )

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, batch_map: Tensor
    ) -> Tensor:
        b = x[:, 0]
        b_max = torch_scatter.scatter(b.abs(), batch_map, reduce="max")
        edge_batch_map = batch_map[edge_index[0]]
        matrix_max = torch_scatter.scatter(
            edge_weight.abs(), edge_batch_map, reduce="max"
        )
        x[:, 0] /= b_max[batch_map]
        x[:, 1] /= matrix_max[batch_map]
        scaled_weights = edge_weight / matrix_max[edge_batch_map]
        y_direction = self.model(x, edge_index, scaled_weights, batch_map)
        return y_direction

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        x, edge_index, edge_weight, batch_map, y, b = batch
        n_systems = batch_map.max().item() + 1
        edge_weight = edge_weight.to(torch.float32)
        b = b.to(torch.float32)
        y = y.to(torch.float32)
        y_direction = self(x, edge_index, edge_weight, batch_map)
        matrix = torch.sparse_coo_tensor(
            edge_index, edge_weight, (b.size(0), b.size(0)), dtype=torch.float32
        )
        b_direction = torch.mv(matrix, y_direction)
        y_loss = self.criterion(y_direction, y, batch_map)
        b_loss = self.criterion(b_direction, b, batch_map)
        loss = y_loss + b_loss
        self.log("loss/train_solution", y_loss, batch_size=n_systems)
        self.log("loss/train_residual", b_loss, batch_size=n_systems)
        self.log("loss/train", loss, batch_size=n_systems)
        return loss

    def _evaluation_step(
        self,
        phase_name: str,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        x, edge_index, edge_weight, batch_map, y, b = batch
        n_systems = batch_map.max().item() + 1
        matrix = torch.sparse_coo_tensor(
            edge_index, edge_weight, (b.size(0), b.size(0)), dtype=torch.float64
        )
        y_direction = self(x, edge_index, edge_weight.to(torch.float32), batch_map)
        y_direction = y_direction.to(torch.float64)
        p_direction = torch.mv(matrix, y_direction)
        p_squared_norm = torch_scatter.scatter_sum(p_direction.square(), batch_map)
        bp_dot_product = torch_scatter.scatter_sum(p_direction * b, batch_map)
        scaler = torch.clamp_min(bp_dot_product / p_squared_norm, 1e-16)
        y_hat = y_direction * scaler[batch_map]
        b_hat = p_direction * scaler[batch_map]
        y_loss = self.criterion(y_hat, y, batch_map)
        b_loss = self.criterion(b_hat, b, batch_map)
        loss = y_loss + b_loss
        self.log(f"loss/{phase_name}_solution", y_loss, batch_size=n_systems)
        self.log(f"loss/{phase_name}_residual", b_loss, batch_size=n_systems)
        self.log(f"loss/{phase_name}", loss, batch_size=n_systems)
        for metric_name, metric in self.systemwise_metrics.items():
            self.log(
                f"metrics/{phase_name}_{metric_name}",
                metric(y_hat, y, batch_map),
                batch_size=n_systems,
            )
            self.log(
                f"residual/{phase_name}_{metric_name}",
                metric(b_hat, b, batch_map),
                batch_size=n_systems,
            )
        self.log(
            f"metrics/{phase_name}_absolute_error",
            self.elementwise_metric(y_hat, y),
            batch_size=y.size(0),
        )

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        self._evaluation_step("val", batch, batch_idx)

    def test_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        self._evaluation_step("test", batch, batch_idx)

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        optimizers = []
        schedulers = []
        optimizer = self.optimizer(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        optimizers.append(optimizer)
        if self.lr_scheduler is not None:
            schedulers.append(
                {
                    "scheduler": self.lr_scheduler(optimizer, **self.scheduler_kwargs),
                    "interval": "epoch",
                    "name": "lr",
                }
            )
        return optimizers, schedulers
