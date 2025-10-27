import torch
import torch.nn.functional as F
from torch import nn

from scipy.linalg import hadamard

from ..utils import FPQuantConfig, FPQuantDtype, validate_config
from .linear_fns import (
    FPQuant4x16MasterFn,
    FPQuant4x4MasterFn,
    FPQuant4x8MasterFn,
    FPQuant4x8NoMasterFn,
    FPQuant4x16NoMasterFn,
    forward_quantize,
)
from .qutlass_ops import HAS_QUTLASS
from .pseudoquant_linear_fns import (
    PseudoQuant4x16MasterFn,
    PseudoQuant4x16NoMasterFn,
    forward_pseudoquantize,
)


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5,
        dtype=dtype,
        device=device,
        requires_grad=False,
    ) * (torch.randint(0, 2, (group_size, 1), device=device, dtype=dtype) * 2 - 1)


def get_identity_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.eye(group_size, dtype=dtype, device=device, requires_grad=False)


def get_gsr_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    hadamard_matrix = get_hadamard_matrix(group_size, dtype, device)
    sign_changes = torch.diff(hadamard_matrix, dim=0).ne(0).sum(dim=0)
    sorted_indices = torch.argsort(sign_changes)
    return hadamard_matrix[:, sorted_indices].contiguous()


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
        validate_config(config)

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
        if self.config.forward_dtype == FPQuantDtype.MXFP4:
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
        elif self.config.forward_dtype == FPQuantDtype.NVFP4:
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
        else:
            raise ValueError(f"Unsupported forward dtype: {config.forward_dtype}")

        # Global scale buffers
        self.register_buffer(
            "weight_global_scale",
            torch.empty(
                1,
                **factory_kwargs,
            ),
        )
        self.register_buffer(
            "act_global_scale",
            torch.empty(
                1,
                **factory_kwargs,
            ),
        )

        # Rotation matrices buffers
        self.register_buffer(
            "forward_hadamard_matrix",
            torch.empty(
                self.config.hadamard_group_size,
                self.config.hadamard_group_size,
                **factory_kwargs,
            ),
        )
        self.register_buffer(
            "backward_hadamard_matrix",
            torch.empty(
                self.config.hadamard_group_size,
                self.config.hadamard_group_size,
                **factory_kwargs,
            ),
        )

    @torch.no_grad()
    def pre_forward(self):
        # Generate rotation matrices
        assert (
            self.weight.shape[1] % self.config.hadamard_group_size == 0
        ), f"Weight shape must be divisible by hadamard group size: {self.weight.shape[1]} % {self.config.hadamard_group_size} = {self.weight.shape[1] % self.config.hadamard_group_size}"

        weight_in_device = self.weight.data.device.type in ["cuda", "xpu"]
        if not self.config.pseudoquantization:
            assert (
                weight_in_device
            ), f"Weight must be on CUDA or XPU, but is on {self.weight.device}"
        if self.config.transform_init == "hadamard":
            transform_init_fn = get_hadamard_matrix
        elif self.config.transform_init == "identity":
            transform_init_fn = get_identity_matrix
        elif self.config.transform_init == "gsr":
            transform_init_fn = get_gsr_matrix
        else:
            raise ValueError(f"Invalid transform init: {self.config.transform_init}")

        self.forward_hadamard_matrix = nn.Buffer(
            transform_init_fn(
                self.config.hadamard_group_size,
                self.weight.dtype,
                self.weight.device,
            ),
        )
        self.backward_hadamard_matrix = nn.Buffer(
            transform_init_fn(
                self.config.hadamard_group_size,
                self.weight.dtype,
                self.weight.device,
            ),
        )

        if (
            self.config.forward_dtype == FPQuantDtype.MXFP4
            and self.config.forward_method == "quest"
        ):
            global_scale_val = 1.0
        elif self.config.forward_method == "abs_max":
            # MXFP4 abs_max quantization implicitly multiplies by 3.0
            global_scale_val = 3.0
        elif self.config.forward_dtype == FPQuantDtype.NVFP4:
            # 10.0 ensures no underflows/overflows in most models
            global_scale_val = 10.0

        self.weight_global_scale = nn.Buffer(
            torch.tensor(
                [global_scale_val],
                dtype=self.weight.dtype,
                device=self.weight.device,
                requires_grad=False,
            ),
        )
        self.act_global_scale = nn.Buffer(
            torch.tensor(
                [global_scale_val],
                dtype=self.weight.dtype,
                device=self.weight.device,
                requires_grad=False,
            ),
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
        if (
            self.config.forward_dtype == FPQuantDtype.MXFP4
            and self.config.backward_dtype == FPQuantDtype.MXFP4
            and self.config.store_master_weights == True
            and self.config.pseudoquantization == False
        ):
            return FPQuant4x4MasterFn.apply(
                x,
                self.weight,
                self.weight_global_scale,
                self.act_global_scale,
                self.bias,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
        elif (
            self.config.forward_dtype == FPQuantDtype.MXFP4
            and self.config.backward_dtype == FPQuantDtype.MXFP8
            and self.config.store_master_weights == True
            and self.config.pseudoquantization == False
        ):
            return FPQuant4x8MasterFn.apply(
                x,
                self.weight,
                self.weight_global_scale,
                self.act_global_scale,
                self.bias,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
        elif (
            self.config.forward_dtype == FPQuantDtype.MXFP4
            and self.config.backward_dtype == FPQuantDtype.MXFP8
            and self.config.store_master_weights == False
            and self.config.pseudoquantization == False
        ):
            return FPQuant4x8NoMasterFn.apply(
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
        elif (
            self.config.forward_dtype in (FPQuantDtype.MXFP4, FPQuantDtype.NVFP4)
            and self.config.backward_dtype == FPQuantDtype.BF16
            and self.config.store_master_weights == True
            and self.config.pseudoquantization == False
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
        elif (
            self.config.forward_dtype in (FPQuantDtype.MXFP4, FPQuantDtype.NVFP4)
            and self.config.backward_dtype == FPQuantDtype.BF16
            and self.config.store_master_weights == False
            and self.config.pseudoquantization == False
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
        elif (
            self.config.forward_dtype in (FPQuantDtype.MXFP4, FPQuantDtype.NVFP4)
            and self.config.backward_dtype == FPQuantDtype.BF16
            and self.config.store_master_weights == True
            and self.config.pseudoquantization == True
        ):
            return PseudoQuant4x16MasterFn.apply(
                x,
                self.weight,
                self.weight_global_scale,
                self.act_global_scale,
                self.bias,
                self.forward_hadamard_matrix,
                self.config.forward_dtype,
                self.config.forward_method,
            )
        elif (
            self.config.forward_dtype in (FPQuantDtype.MXFP4, FPQuantDtype.NVFP4)
            and self.config.backward_dtype == FPQuantDtype.BF16
            and self.config.store_master_weights == False
            and self.config.pseudoquantization == True
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
        else:
            raise ValueError(
                f"Forward dtype: {self.config.forward_dtype}, backward dtype: {self.config.backward_dtype}, store_master_weights: {self.config.store_master_weights}, pseudoquantization: {self.config.pseudoquantization} isn't supported yet."
            )
