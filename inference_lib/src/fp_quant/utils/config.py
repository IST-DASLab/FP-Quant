from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class FPQuantDtype(Enum):
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"
    BF16 = "bf16"


QuantMethod = Literal["quest", "abs_max"]


@dataclass
class FPQuantConfig:
    forward_dtype: FPQuantDtype = FPQuantDtype.MXFP4
    forward_method: QuantMethod = "quest"
    backward_dtype: FPQuantDtype = FPQuantDtype.MXFP4
    store_master_weights: bool = False
    hadamard_group_size: int = 32
    pseudoquantization: bool = False
    modules_to_not_convert: list[str] = field(default_factory=lambda: ["lm_head"])
