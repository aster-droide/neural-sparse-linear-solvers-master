import torch

from torch import Tensor
from torch_scatter import scatter_sum


class CosineDistanceLoss(torch.nn.Module):
    def forward(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> Tensor:
        eps = 1e-12
        preds_norm = scatter_sum(preds.square(), batch_map).sqrt_()
        target_norm = scatter_sum(target.square(), batch_map).sqrt_()
        dot_product = scatter_sum(preds * target, batch_map)
        cosine = dot_product / torch.clamp_min(preds_norm * target_norm, eps)
        cosine_distance = 1 - cosine
        return cosine_distance.mean()


class L1DistanceLoss(torch.nn.Module):
    def forward(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> Tensor:
        absolute_difference = torch.abs(preds - target)
        l1_distance = scatter_sum(absolute_difference, batch_map)
        return l1_distance.mean()


class L2DistanceLoss(torch.nn.Module):
    def forward(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> Tensor:
        squared_difference = torch.square(preds - target)
        l2_distance = scatter_sum(squared_difference, batch_map).sqrt_()
        return l2_distance.mean()
