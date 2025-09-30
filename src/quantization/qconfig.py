from typing import Any

def prepare_quantization_config(
    hadamard_group_size: int, 
    format: str,
    pseudoquantization: bool = False
) -> dict[str, Any]:
    if format in ["mxfp", "nvfp"]:
        return {
            "forward_dtype": f"{format}4",
            "backward_dtype": "bf16",
            "forward_method": "abs_max",
            "hadamard_group_size":hadamard_group_size,
            "modules_to_not_convert": ["lm_head"],
            "quant_method": "fp_quant",
            "store_master_weights": False,
            "pseudoquantization": pseudoquantization
        }
    else:
        raise ValueError(f"Invalid format: {format}")
