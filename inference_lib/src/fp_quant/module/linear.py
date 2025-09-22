import torch
import torch.nn.functional as F
from torch import nn

from scipy.linalg import hadamard

from ..utils import FPQuantConfig, FPQuantDtype

from . import layer_analytics

from .linear_fns import (
    HAS_QUTLASS,
    FPQuant4x16MasterFn,
    FPQuant4x16NoMasterFn,
    forward_quantize,
)
from .pseudoquant_linear_fns import (
    PseudoQuant4x16MasterFn,
    PseudoQuant4x16NoMasterFn,
    forward_pseudoquantize,
)


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5, dtype=dtype, device=device
    )

def free_param(param: torch.nn.Parameter):
    if param is not None:
        # Move off GPU to free CUDA memory
        if param.device.type == "cuda":
            param.data = param.data.cpu()

        # Break the computational graph and drop storage reference
        param.detach_()

        # Remove the parameter reference
        del param

class FPQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: FPQuantConfig,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        name: str = None,
        enable_analytics: bool = False,
    ):
        super().__init__()

        if not HAS_QUTLASS and not config.pseudoquantization:
            raise ValueError(
                "QuTLASS is not installed. Can only run with `pseudoquantization=True` in the quantization config. If you have a Blackwell GPU, you can install QuTLASS from https://github.com/IST-DASLab/QuTLASS"
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.device = device
        self.dtype = dtype
        
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.name_analyzed = not enable_analytics

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.dqweight = None

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.config = config

        # Quantized tensors buffers
        match self.config.forward_dtype:
            case FPQuantDtype.MXFP4:
                self.register_buffer(
                    "qweight",
                    torch.empty(
                        self.weight.shape[0],
                        self.weight.shape[1] // 2,
                        dtype=torch.uint8,
                        device=self.weight.device,
                    ),
                )
            case FPQuantDtype.MXFP8:
                self.register_buffer(
                    "qweight",
                    torch.empty(
                        *self.weight.shape, dtype=torch.uint8, device=self.weight.device
                    ),
                )
            case _:
                raise ValueError(f"Unsupported forward dtype: {config.forward_dtype}")
        self.register_buffer(
            "scales",
            torch.empty(
                self.weight.shape[0],
                self.weight.shape[1] // 32,
                dtype=torch.uint8,
                device=self.weight.device,
            ),
        )

        # Rotation matrices buffers
        self.register_buffer(
            "forward_hadamard_matrix",
            torch.empty(
                self.config.hadamard_group_size,
                self.config.hadamard_group_size,
                dtype=self.weight.dtype,
                device=self.weight.device,
            ),
        )
        self.register_buffer(
            "backward_hadamard_matrix",
            torch.empty(
                self.config.hadamard_group_size,
                self.config.hadamard_group_size,
                dtype=self.weight.dtype,
                device=self.weight.device,
            ),
        )

    @torch.no_grad()
    def pre_forward(self):
        # Generate rotation matrices
        assert self.weight.shape[1] % self.config.hadamard_group_size == 0, (
            f"Weight shape must be divisible by hadamard group size: {self.weight.shape[1]} % {self.config.hadamard_group_size} = {self.weight.shape[1] % self.config.hadamard_group_size}"
        )
        assert self.weight.data.is_cuda, (
            f"Weight must be on CUDA, but is on {self.weight.device}"
        )
        self.forward_hadamard_matrix = nn.Parameter(
            get_hadamard_matrix(
                self.config.hadamard_group_size,
                self.weight.dtype,
                self.weight.device,
            )
        )
        self.backward_hadamard_matrix = nn.Parameter(
            get_hadamard_matrix(
                self.config.hadamard_group_size,
                self.weight.dtype,
                self.weight.device,
            )
        )

        # Quantize weights
        if self.config.store_master_weights:
            self.qweight = None
            self.scales = None
            self.dqweight = None
        elif self.config.pseudoquantization:
            weight_dq, _ = forward_pseudoquantize(
                self.weight.data,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
            self.dqweight = nn.Parameter(weight_dq, requires_grad=False)
            self.weight = None
            self.qweight = None
            self.scales = None
        else:
            weight_q, scales, _ = forward_quantize(
                self.weight,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
            self.qweight = nn.Parameter(weight_q, requires_grad=False)
            self.scales = nn.Parameter(
                scales.view(dtype=torch.uint8), requires_grad=False
            )

            if self.weight is not None:
                free_param(self.weight)
                self.register_parameter("weight", None)
                torch.cuda.empty_cache()
                del self.weight

            self.weight = None
            self.dqweight = None

    def forward(self, x) -> torch.Tensor:
        
        if self.name is not None and not self.name_analyzed:
            self.name_analyzed = True
            layer_analytics.add_layer(self, self.name, x.shape, self.in_features, self.out_features, x.device, x.dtype)


        result = None

        match (
            self.config.forward_dtype,
            self.config.backward_dtype,
            self.config.store_master_weights,
            self.config.pseudoquantization,
        ):
            case (FPQuantDtype.MXFP4, FPQuantDtype.BF16, True, False):
                result = FPQuant4x16MasterFn.apply(
                    x,
                    self.weight,
                    self.bias,
                    self.forward_hadamard_matrix,
                    self.config.forward_dtype,
                    self.config.forward_method,
                )
            case (FPQuantDtype.MXFP4, FPQuantDtype.BF16, False, False):
                result = FPQuant4x16NoMasterFn.apply(
                    x,
                    self.qweight,
                    self.scales,
                    self.bias,
                    self.forward_hadamard_matrix,
                    self.config.forward_dtype,
                    self.config.forward_method,
                )
            case (FPQuantDtype.MXFP4, FPQuantDtype.BF16, True, True):
                result = PseudoQuant4x16MasterFn.apply(
                    x,
                    self.dqweight,
                    self.bias,
                    self.forward_hadamard_matrix,
                    self.config.forward_dtype,
                    self.config.forward_method,
                )
            case (FPQuantDtype.MXFP4, FPQuantDtype.BF16, False, True):
                result = PseudoQuant4x16NoMasterFn.apply(
                    x,
                    self.dqweight,
                    self.bias,
                    self.forward_hadamard_matrix,
                    self.config.forward_dtype,
                    self.config.forward_method,
                )
            case _:
                raise ValueError(
                    f"Forward dtype: {self.config.forward_dtype}, backward dtype: {self.config.backward_dtype}, store_master_weights: {self.config.store_master_weights}, pseudoquantization: {self.config.pseudoquantization} isn't supported yet."
                )
            
        return result
    