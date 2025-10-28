import torch
from torch import nn

from .config import FPQuantConfig


def replace_with_fp_quant_linear(
    model,
    fp_quant_linear_config: FPQuantConfig,
    current_key_name=None,
    has_been_replaced=False,
):
    from ..module import FPQuantLinear

    """
    Public method that recursively replaces the Linear layers of the given model with HIGGS quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successful or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`HiggsConfig`):
            The quantization config object that contains the quantization parameters.
        current_key_name (`list`, *optional*):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """

    from accelerate import init_empty_weights

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear):
            # Check if the current key is not in the `quantization_config.modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                current_key_name_str.endswith(key)
                for key in fp_quant_linear_config.modules_to_not_convert
            ):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features

                    model._modules[name] = FPQuantLinear(
                        in_features,
                        out_features,
                        config=fp_quant_linear_config,
                        bias=module.bias is not None,
                    )
                    has_been_replaced = True

                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_fp_quant_linear(
                module,
                fp_quant_linear_config=fp_quant_linear_config,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_quantize_with_fp_quant_linear(
    model,
    fp_quant_linear_config: FPQuantConfig,
    current_key_name=None,
    has_been_replaced=False,
):
    from ..module import FPQuantLinear

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear):
            # Check if the current key is not in the `quantization_config.modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                current_key_name_str.endswith(key)
                for key in fp_quant_linear_config.modules_to_not_convert
            ):
                in_features = module.in_features
                out_features = module.out_features

                model._modules[name] = FPQuantLinear(
                    in_features,
                    out_features,
                    config=fp_quant_linear_config,
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                model._modules[name].weight.data = module.weight.data
                if module.bias is not None:
                    model._modules[name].bias.data = module.bias.data
                model._modules[name].pre_forward()
        if len(list(module.children())) > 0:
            _ = replace_quantize_with_fp_quant_linear(
                module,
                fp_quant_linear_config=fp_quant_linear_config,
                current_key_name=current_key_name,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model


def finalize_master_weights(
    model,
    current_key_name=None,
):
    from ..module import FPQuantLinear

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, FPQuantLinear):
            if model._modules[name].config.store_master_weights:
                model._modules[name].config.store_master_weights = (
                    False  # all FPQuantLinear share the same config obj
                )
            model._modules[name].pre_forward()

        if len(list(module.children())) > 0:
            finalize_master_weights(
                model._modules[name],
                current_key_name=current_key_name,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
