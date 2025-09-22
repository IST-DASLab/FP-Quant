import gc
import torch
from torch import nn

from .config import FPQuantConfig

from ..module import layer_analytics 

def replace_with_fp_quant_linear(
    model,
    fp_quant_linear_config: FPQuantConfig,
    current_key_name=None,
    has_been_replaced=False,
    apply_pre_forward=False,
    enable_analytics=False,
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
    num_of_params = 0

    for name, module in model.named_children():

        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear):
            # Check if the current key is not in the `quantization_config.modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                current_key_name_str.startswith(key)
                for key in fp_quant_linear_config.modules_to_not_convert
            ):
                in_features = module.in_features
                out_features = module.out_features

                layer_analytics.all_added_layer_names.append(current_key_name_str)

                new = FPQuantLinear(in_features, out_features, config=fp_quant_linear_config, bias=module.bias is not None, device=module.weight.device, dtype=module.weight.dtype, name=current_key_name_str, enable_analytics=enable_analytics)
                with torch.no_grad():
                    if hasattr(new, "load_from_linear"):
                        new.load_from_linear(module)  # hypothetical helper
                    else:
                        # fallback if FPQuantLinear stores real-valued weights
                        if hasattr(new, "weight") and hasattr(module, "weight"):
                            new.weight.copy_(module.weight)
                        if hasattr(new, "bias") and hasattr(module, "bias") and module.bias is not None:
                            new.bias.copy_(module.bias)
                
                    model._modules[name] = new
                    
                    module.weight.to(device='cpu')
                    if module.bias is not None:
                        module.bias.to(device='cpu')

                    has_been_replaced = True

                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)

                    # Force delete the tensors here
                    if hasattr(module, "weight") and module.weight is not None:
                        num_of_params += module.weight.numel()
                        del module.weight
                        module._parameters.pop("weight", None)
                        module.register_parameter("weight", None)

                    if hasattr(module, "bias") and module.bias is not None:
                        del module.bias
                        module._parameters.pop("bias", None)
                        module.register_parameter("bias", None)

                    del module  

                    if apply_pre_forward:
                        model._modules[name].pre_forward()

                torch.cuda.empty_cache()

        elif len(list(module.children())) > 0:
            _, has_been_replaced, num_of_params_temp = replace_with_fp_quant_linear(
                module,
                fp_quant_linear_config=fp_quant_linear_config,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
                apply_pre_forward=apply_pre_forward,
                enable_analytics=enable_analytics
            )
            num_of_params += num_of_params_temp

        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced, num_of_params
