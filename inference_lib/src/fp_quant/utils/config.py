from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal


class FPQuantDtype(Enum):
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"
    BF16 = "bf16"


QuantMethod = Literal["quest", "abs_max"]
TransformInit = Literal["hadamard", "identity", "gsr"]


@dataclass
class FPQuantConfig:
    forward_dtype: FPQuantDtype = FPQuantDtype.MXFP4
    forward_method: QuantMethod = "quest"
    backward_dtype: FPQuantDtype = FPQuantDtype.MXFP4
    store_master_weights: bool = False
    hadamard_group_size: int = 32
    pseudoquantization: bool = False
    transform_init: TransformInit = "hadamard"
    modules_to_not_convert: List[str] = field(default_factory=lambda: ["lm_head"])


def validate_config(config: FPQuantConfig):
    if (
        config.forward_dtype == FPQuantDtype.NVFP4
        and config.forward_method != "abs_max"
    ):
        raise ValueError("NVFP4 can only be used with abs_max method")
    if (
        config.forward_dtype == FPQuantDtype.NVFP4
        and config.hadamard_group_size not in [16, 32, 64, 128]
    ):
        raise ValueError(
            "NVFP4 can only be used with hadamard_group_size in [16, 32, 64, 128]"
        )
    if (
        config.forward_dtype == FPQuantDtype.MXFP4
        and config.hadamard_group_size not in [32, 64, 128]
    ):
        raise ValueError(
            "MXFP4 can only be used with hadamard_group_size in [32, 64, 128]"
        )
