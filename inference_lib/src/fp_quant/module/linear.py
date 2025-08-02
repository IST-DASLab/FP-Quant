import torch
import torch.nn.functional as F
from torch import nn

from scipy.linalg import hadamard

from ..utils import FPQuantConfig, FPQuantDtype
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


class FPQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: FPQuantConfig,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        if not HAS_QUTLASS and not config.pseudoquantization:
            raise ValueError(
                "QuTLASS is not installed. Can only run with `pseudoquantization=True` in the quantization config. If you have a Blackwell GPU, you can install QuTLASS from https://github.com/IST-DASLab/QuTLASS"
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.dqweight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
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
                self.register_buffer(
                    "scales",
                    torch.empty(
                        self.weight.shape[0],
                        self.weight.shape[1] // 32,
                        dtype=torch.uint8,
                        device=self.weight.device,
                    ),
                )
            case FPQuantDtype.NVFP4:
                self.register_buffer(
                    "qweight",
                    torch.empty(
                        self.weight.shape[0],
                        self.weight.shape[1] // 2,
                        dtype=torch.uint8,
                        device=self.weight.device,
                    ),
                )
                self.register_buffer(
                    "scales",
                    torch.empty(
                        self.weight.shape[0],
                        self.weight.shape[1] // 16,
                        dtype=torch.uint8,
                        device=self.weight.device,
                    ),
                )
            case _:
                raise ValueError(f"Unsupported forward dtype: {config.forward_dtype}")

        # Global scale buffers
        self.register_buffer(
            "weight_global_scale",
            torch.empty(
                1,
                dtype=torch.float32,
                device=self.weight.device,
            ),
        )
        self.register_buffer(
            "act_global_scale",
            torch.empty(
                1,
                dtype=torch.float32,
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

        if self.config.forward_dtype == FPQuantDtype.MXFP4:
            # MXFP4 quantization implicitly multiplies by 3.0
            self.weight_global_scale = nn.Parameter(
                torch.tensor(
                    [1.0 / 3.0], dtype=torch.float32, device=self.weight.device
                )
            )
            self.act_global_scale = nn.Parameter(
                torch.tensor(
                    [1.0 / 3.0], dtype=torch.float32, device=self.weight.device
                )
            )
        elif self.config.forward_dtype == FPQuantDtype.NVFP4:
            # MXFP4 quantization implicitly multiplies by 6.0
            self.weight_global_scale = nn.Parameter(
                torch.tensor(
                    [1.0 / 6.0], dtype=torch.float32, device=self.weight.device
                )
            )
            self.act_global_scale = nn.Parameter(
                torch.tensor(
                    [1.0 / 6.0], dtype=torch.float32, device=self.weight.device
                )
            )

        if self.config.store_master_weights:
            self.qweight = None
            self.scales = None
            self.dqweight = None
        elif self.config.pseudoquantization:
            weight_dq, _ = forward_pseudoquantize(
                self.weight.data,
                self.forward_hadamard_matrix,
                self.weight_global_scale,
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
                self.weight_global_scale,
                self.config.forward_dtype,
                self.config.forward_method,
            )
            self.qweight = nn.Parameter(weight_q, requires_grad=False)
            self.scales = nn.Parameter(
                scales.view(dtype=torch.uint8), requires_grad=False
            )
            self.weight = None
            self.dqweight = None

    def forward(self, x) -> torch.Tensor:
        match (
            self.config.forward_dtype,
            self.config.backward_dtype,
            self.config.store_master_weights,
            self.config.pseudoquantization,
        ):
            case (
                FPQuantDtype.MXFP4 | FPQuantDtype.NVFP4,
                FPQuantDtype.BF16,
                True,
                False,
            ):
                return FPQuant4x16MasterFn.apply(
                    x,
                    self.weight,
                    self.weight_global_scale,
                    self.act_global_scale,
                    self.bias,
                    self.forward_hadamard_matrix,
                    self.config.forward_dtype,
                    self.config.forward_method,
                )
            case (
                FPQuantDtype.MXFP4 | FPQuantDtype.NVFP4,
                FPQuantDtype.BF16,
                False,
                False,
            ):
                return FPQuant4x16NoMasterFn.apply(
                    x,
                    self.qweight,
                    self.scales,
                    self.weight_global_scale,
                    self.act_global_scale,
                    self.bias,
                    self.forward_hadamard_matrix,
                    self.config.forward_dtype,
                    self.config.forward_method,
                )
            case (
                FPQuantDtype.MXFP4 | FPQuantDtype.NVFP4,
                FPQuantDtype.BF16,
                True,
                True,
            ):
                return PseudoQuant4x16MasterFn.apply(
                    x,
                    self.dqweight,
                    self.weight_global_scale,
                    self.act_global_scale,
                    self.bias,
                    self.forward_hadamard_matrix,
                    self.config.forward_dtype,
                    self.config.forward_method,
                )
            case (
                FPQuantDtype.MXFP4 | FPQuantDtype.NVFP4,
                FPQuantDtype.BF16,
                False,
                True,
            ):
                return PseudoQuant4x16NoMasterFn.apply(
                    x,
                    self.dqweight,
                    self.weight_global_scale,
                    self.act_global_scale,
                    self.bias,
                    self.forward_hadamard_matrix,
                    self.config.forward_dtype,
                    self.config.forward_method,
                )
            case _:
                raise ValueError(
                    f"Forward dtype: {self.config.forward_dtype}, backward dtype: {self.config.backward_dtype}, store_master_weights: {self.config.store_master_weights}, pseudoquantization: {self.config.pseudoquantization} isn't supported yet."
                )
