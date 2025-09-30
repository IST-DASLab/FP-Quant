import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import Quantizer
from ..transforms.transforms import BaseTransform


class QLinear(nn.Linear):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        weight_quantizer: Quantizer = None,
        act_quantizer: Quantizer = None,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.weight_quantizer = weight_quantizer
        self.act_quantizer = act_quantizer
        self._train_mode = True

    def forward(
        self, 
        x: torch.Tensor, 
        in_transform: BaseTransform = None, 
        out_transform: BaseTransform = None
    ) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if self._train_mode:
            if in_transform:
                weight = in_transform(weight, inv_t=True, dim=-1)
            if out_transform:
                weight = out_transform(weight, inv_t=True, dim=0)
                if bias is not None:
                    bias = out_transform(bias, inv_t=True, dim=0)

            if self.weight_quantizer is not None:
                w_scales, w_zeros = self.weight_quantizer.get_quantization_params(weight)
                weight = self.weight_quantizer(weight, w_scales, w_zeros)

        if self.act_quantizer is not None:
            a_scales, a_zeros = self.act_quantizer.get_quantization_params(x)
            x = self.act_quantizer(x, a_scales, a_zeros)

        return F.linear(x, weight, bias)

    def fix_parametrization(
        self, 
        in_transform: BaseTransform = None, 
        out_transform: BaseTransform = None
    ) -> None:
        weight = self.weight
        bias = self.bias

        if in_transform:
            weight = in_transform(weight, inv_t=True, dim=-1)
        if out_transform:
            weight = out_transform(weight, inv_t=True, dim=0)
            if bias is not None:
                bias = out_transform(bias, inv_t=True, dim=0)

        if self.weight_quantizer is not None:
            w_scales, w_zeros = self.weight_quantizer.get_quantization_params(weight)
            weight = self.weight_quantizer(weight, w_scales, w_zeros)
            self.weight_quantizer._track_global_scale = False

        if self.act_quantizer is not None:
            self.act_quantizer._track_global_scale = False

        self.weight.data = weight
        if bias is not None:
            self.bias.data = bias

        self._train_mode = False
