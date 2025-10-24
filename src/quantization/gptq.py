import re
import gc
import math
import argparse
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from transformers import AutoModelForCausalLM

from .qlinear import QLinear
from .quantizer import Quantizer
from .quant_args import QuantizationOrder
from .quant_ops import pack_fp4_to_uint8, cast_scales_to_eXmY, ScalePrecision
from .accumulate_hessian import accumulate_hessian
from ..transforms.transforms import build_transform, get_transform_matrix
from ..utils.linalg_utils import inv_sym
from ..utils.common_utils import clear_device_cache, to, maybe_first_element
from ..utils.model_utils import InputCollector, ForwardInterrupt, get_attention_layer, get_mlp_layer, get_number_of_rows_and_cols

try:
    import wandb
except ImportError:
    wandb = None


def get_relative_mse_error(q: torch.Tensor, w: torch.Tensor, H: torch.Tensor):
    delta = q - w
    return (delta).mm(H).mul(delta).mean() / (w.mm(H).mul(w).mean() + 1e-6)


class GPTQ:

    def __init__(
        self,
        layer: nn.Module,
        quantizer: Quantizer,
        quantization_order: str = "default",
        block_size: int = 128,
        rel_damp: float = 1e-2,
        export_quantized_model: str = "",
    ):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = get_number_of_rows_and_cols(layer)
        # Quantization properties
        self.quantizer = quantizer
        self.quantization_order = QuantizationOrder(quantization_order)
        self.block_size = block_size
        self.rel_damp = rel_damp
        # Backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.H = None
        self.num_samples = 0
        # Whether to apply real quantization
        self.export_quantized_model = export_quantized_model

    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "OBC supports only linear and convolutional layers."

    # preparatory methods
    @torch.no_grad()
    def update(self, input: torch.Tensor) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            input: batch of layer inputs
        """
        # get batch size
        batch_size = input.shape[0]
        # init hessian
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        # input reshaping
        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            # output size (batch_size, channels * \prod kernel_size, num_patches)
            input = unfold(input)
            input = input.transpose(1, 2).flatten(0, 1)
        # cast input to float32 before addition
        input = input.float()
        # rescale and update matrix
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.mul_(beta)
        input.mul_(math.sqrt(alpha))
        accumulate_hessian(self.H, input)
        self.num_samples += batch_size

    def reset(self) -> None:
        self.W = self.layer.weight
        self.H = None
        self.num_samples = 0
        clear_device_cache()

    @torch.no_grad()
    def quantization_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        # 1) Hessian preparation
        assert self.H is not None, "One has to process at least one sample of calibration data to run pruning"
        # 2) Weight preparation
        # copy weight, flatten and convert to float
        self.W = self.W.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)
        # flag pre step as completed
        self.pre_step_completed = True

    @torch.no_grad()
    def step(self) -> torch.Tensor | Optional[torch.Tensor] | torch.Tensor:
        """
        Quantize the weight matrix using GPTQ
        """
        # 1) Define constants and chunk
        d_col, block_size, device, dtype = self.d_col, self.block_size, self.W_device, self.W_dtype
        # 2) Get quantization group size
        quantizer_group_size = self.quantizer.group_size
        group_size = quantizer_group_size or d_col
        num_groups = d_col // group_size

        # Init quantized weight
        qweight = None
        if self.export_quantized_model:
            qweight = torch.empty(self.W.shape, device=device, dtype=dtype)
        # Get scales and zeros 
        scales, zeros = self.quantizer.get_quantization_params(self.W) 
        # Dirty hack for GPTQ quantization
        self.quantizer.group_size = None
        # Get permutation
        if self.quantization_order == QuantizationOrder.ACTIVATION:
            perm = torch.argsort(self.H.diag(), descending=True)
            group_idx = torch.arange(num_groups, device=device).repeat_interleave(group_size)[perm]
        else:
            perm = torch.arange(d_col, device=device)
        perm_inv = torch.argsort(perm)
        # Permute Hessian prior to inversion
        self.H = self.H[perm][:, perm]
        # Get weight
        w = self.W[:, perm]
        # Get Hessian inverse   
        H_inv_cho = self._get_hessian_inverse()
        # Quantize
        for c1 in range(0, d_col, block_size):
            c2 = min(c1 + block_size, d_col)
            ncols = c2 - c1
            w_blk = w[:, c1:c2].clone()  
            errs = torch.zeros_like(w_blk)
            H_inv_cho_blk = H_inv_cho[c1:c2, c1:c2]
            # 2) Iterate over block
            for i in range(ncols):
                # Get weight column, corresponding Hessian diagonal and group_id
                w_ci = w_blk[:, i]
                d = H_inv_cho_blk[i, i]
                if self.quantization_order == QuantizationOrder.ACTIVATION:
                    g_idx = group_idx[c1 + i]
                else:
                    g_idx = (c1 + i) // group_size    
                # Quantize weight column
                if self.export_quantized_model:
                    q = self.quantizer.quantize(w_ci, scales[:, g_idx], zeros[:, g_idx])
                    w_q = self.quantizer.dequantize(q, scales[:, g_idx], zeros[:, g_idx])
                    qweight[:, c1 + i] = q
                else:
                    w_q = self.quantizer(w_ci, scales[:, g_idx], zeros[:, g_idx])
                w[:, c1 + i] = w_q
                # Update subsequent weight
                err = (w_ci - w_q) / d
                w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
                errs[:, i] = err
            # 3) Update the weights after block
            w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

        # Invert permutation
        w = w[:, perm_inv].contiguous()
        if qweight is not None:
            qweight = qweight[:, perm_inv].contiguous()
        self.H = self.H[perm_inv][:, perm_inv]
        # Restore quantizer group size
        self.quantizer.group_size = quantizer_group_size
        
        return w.to(dtype), qweight, scales
    
    @torch.no_grad()
    def _get_hessian_inverse(self):
        w = self.W
        # Get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        H = self.H
        # mask rows with zero input channels
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        # Hessian regularization
        damp = self.rel_damp * torch.diag(self.H).mean()
        self.H[range(self.d_col), range(self.d_col)] += damp
        # invert
        try:
            H = inv_sym(H)
            H_inv_cho = torch.linalg.cholesky(H, upper=True)
        except:
            H_inv_cho = torch.eye(self.d_col, device=H.device, dtype=torch.float32)
        return H_inv_cho

    def quantize(self) -> torch.Tensor | Optional[torch.Tensor] | torch.Tensor:
        self.quantization_pre_step()
        return self.step()


def gptq_quantization(
    model: AutoModelForCausalLM, 
    calibration_data: List[torch.Tensor],
    args: argparse.Namespace, 
    device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    print("GPTQ quantization...")
    orig_dtype = model.config.torch_dtype if args.dtype == "auto" else args.dtype
    act_offload_device = "cpu" if args.cpu_offload_activations else device
    # State dict with quantized weights, scales and hadamards
    quantized_state_dict = {}
    non_quantized_state_dict = {}
    # Define common transform kwargs
    transform_kwargs = dict(device=device, group_size=args.hadamard_group_size)
    # Init quantizer kwargs
    weight_quantizer_kwargs = None
    if args.w_bits < 16:
        weight_quantizer_kwargs = dict(
            bits=args.w_bits, 
            symmetric=True, 
            format=args.format,
            granularity=args.w_granularity,
            observer=args.w_observer, 
            group_size=args.w_group_size,
            scale_precision=args.scale_precision
        )
    act_quantizer_kwargs = None
    if args.a_bits < 16:
        act_quantizer_kwargs = dict(
            bits=args.a_bits,
            symmetric=True, 
            format=args.format,
            granularity=args.a_granularity,
            observer=args.a_observer, 
            group_size=args.a_group_size,
            scale_precision=args.scale_precision
        )

    blocks = model.model.layers
    blocks[0] = InputCollector(blocks[0], cpu_offload=args.cpu_offload_activations)
    if args.cpu_offload_modules:
        model.get_input_embeddings().to(device)
        blocks[0] = blocks[0].to(device)

    for sample in calibration_data:
        try:
            with torch.no_grad():
                model(sample.to(device=device))
        except ForwardInterrupt:
            if args.cpu_offload_activations:
                sample = sample.to(device="cpu")
            pass
        
    input_args = blocks[0].input_args
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module

    if args.cpu_offload_modules:
        model.get_input_embeddings().cpu()

    # Iterate over transformer blocks
    for block_idx, block in enumerate(blocks):
        print(f"Processing block {block_idx}...")
        if args.cpu_offload_modules:
            block.to(device)

        # 1. Init transforms
        qkv_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        o_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        gate_up_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        down_in_transform = build_transform(args.transform_class, size=model.config.intermediate_size, **transform_kwargs)     

        # 2. Replace blocks with quantized versions
        quantized_attn = get_attention_layer(model.config)(
            model.config,
            layer_idx=block_idx,
            act_quantizer_kwargs=act_quantizer_kwargs,
            qkv_in_transform=qkv_in_transform,
            o_in_transform=o_in_transform
        )
        quantized_mlp = get_mlp_layer(model.config)(
            model.config,
            act_quantizer_kwargs=act_quantizer_kwargs,
            gate_up_in_transform=gate_up_in_transform,
            down_in_transform=down_in_transform
        )

        quantized_attn.load_state_dict(block.self_attn.state_dict(), strict=False)
        quantized_mlp.load_state_dict(block.mlp.state_dict(), strict=False)

        block.self_attn = quantized_attn
        block.mlp = quantized_mlp

        # Move to original device and dtype
        block = block.to(device=device, dtype=orig_dtype)
        # Toggle off gradients for all parameters
        block.requires_grad_(False)

        # 3. Fix transforms and remove parametrizations
        qkv_in_transform.remove_parametrizations()
        o_in_transform.remove_parametrizations()
        gate_up_in_transform.remove_parametrizations()
        down_in_transform.remove_parametrizations() 

        # 4. Create GPTQ handles and hooks
        gptq_handles = {}
        hooks = {}
        for layer_name, layer in block.named_modules():
            if isinstance(layer, QLinear):
                # Create GPTQ handle
                gptq_handles[layer_name] = GPTQ(
                    layer, 
                    Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None, 
                    quantization_order=args.quantization_order, 
                    rel_damp=args.rel_damp,
                    export_quantized_model=args.export_quantized_model
                )
                # Get weight global scale
                if args.scale_precision == ScalePrecision.E4M3:
                    # Rotate weight
                    with torch.no_grad():
                        if re.search("(q|k|v)_proj", layer_name):
                            layer_transform = qkv_in_transform
                        elif re.search("o_proj", layer_name):
                            layer_transform = o_in_transform
                        elif re.search("(gate|up)_proj", layer_name):
                            layer_transform = gate_up_in_transform
                        else:
                            layer_transform = down_in_transform
                        weight = layer_transform(layer.weight, inv_t=True)
                    
                    gptq_handles[layer_name].quantizer.get_quantization_params(weight)
                    # Turn off global scale tracking
                    gptq_handles[layer_name].quantizer._track_global_scale = False
                # Attach hook
                def update_handle_hook(name):
                    def _hook(_, inp, out):
                        gptq_handles[name].update(inp[0])
                    return _hook
                hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))

        # Fuse global scales
        if args.fuse_global_scale and args.scale_precision == ScalePrecision.E4M3:
            # qkv fusion
            qkv_global_scale = min(
                gptq_handles["self_attn.q_proj"].quantizer.global_scale,
                gptq_handles["self_attn.k_proj"].quantizer.global_scale,
                gptq_handles["self_attn.v_proj"].quantizer.global_scale,
            )
            gptq_handles["self_attn.q_proj"].quantizer.global_scale = qkv_global_scale
            gptq_handles["self_attn.k_proj"].quantizer.global_scale = qkv_global_scale
            gptq_handles["self_attn.v_proj"].quantizer.global_scale = qkv_global_scale
            # gate_up fusion
            gate_up_global_scale = min(
                gptq_handles["mlp.gate_proj"].quantizer.global_scale,
                gptq_handles["mlp.up_proj"].quantizer.global_scale
            )
            gptq_handles["mlp.gate_proj"].quantizer.global_scale = gate_up_global_scale
            gptq_handles["mlp.up_proj"].quantizer.global_scale = gate_up_global_scale

        # 5. Process calibration data
        device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
        # Remove hooks
        for hook in hooks.values():
            hook.remove()

        # 6. Transform all weights before quantization
        block.self_attn.q_proj.weight.data = qkv_in_transform(block.self_attn.q_proj.weight, inv_t=True)
        block.self_attn.k_proj.weight.data = qkv_in_transform(block.self_attn.k_proj.weight, inv_t=True)
        block.self_attn.v_proj.weight.data = qkv_in_transform(block.self_attn.v_proj.weight, inv_t=True)
        block.self_attn.o_proj.weight.data = o_in_transform(block.self_attn.o_proj.weight, inv_t=True)
        block.mlp.gate_proj.weight.data = gate_up_in_transform(block.mlp.gate_proj.weight, inv_t=True)
        block.mlp.up_proj.weight.data = gate_up_in_transform(block.mlp.up_proj.weight, inv_t=True)
        block.mlp.down_proj.weight.data = down_in_transform(block.mlp.down_proj.weight, inv_t=True)
        # Set train_mode to False
        for layer_name, layer in block.named_modules():
            if isinstance(layer, QLinear):
                layer._train_mode = False
                if layer.act_quantizer:
                    layer.act_quantizer._track_global_scale = False

        # 7. Run GPTQ quantization
        for layer_name, gptq_handle in gptq_handles.items():
            dequantized_qweight, qweight, scales = gptq_handle.quantize()
            orig_weight = gptq_handle.layer.weight
            with torch.no_grad():
                relative_mse_error = get_relative_mse_error(dequantized_qweight.float(), orig_weight.float(), gptq_handle.H)
            print(f"[{layer_name:16}]: Relative MSE error: {relative_mse_error.item():.2e}")
            if args.log_wandb:
                wandb.log({f"gptq/{layer_name}_relative_mse": relative_mse_error.item()})
            gptq_handle.layer.weight.data = dequantized_qweight
            
            # Update quantized state dict (if needed)
            if args.export_quantized_model:
                weight_global_scale = gptq_handle.quantizer.global_scale.to(scales.device)
                act_global_scale = gptq_handle.layer.act_quantizer.global_scale

                transform_matrix = get_transform_matrix(args.transform_class, args.hadamard_group_size, device, orig_dtype).cpu()

                if args.export_quantized_model == "realquant":
                    quantized_state_dict[f"model.layers.{block_idx}.{layer_name}"] = {
                        "qweight": pack_fp4_to_uint8(qweight).cpu(),
                        "scales": cast_scales_to_eXmY(scales * weight_global_scale, args.scale_precision).cpu(),
                        "forward_hadamard_matrix": transform_matrix,
                        "backward_hadamard_matrix": transform_matrix.clone(),
                        "weight_global_scale": weight_global_scale.clone(),
                        "act_global_scale": act_global_scale.clone()
                    }
                # pseudoquant
                else:
                    quantized_state_dict[f"model.layers.{block_idx}.{layer_name}"] = {
                        "dqweight": dequantized_qweight.cpu(),
                        "forward_hadamard_matrix": transform_matrix,
                        "backward_hadamard_matrix": transform_matrix.clone(),
                        "weight_global_scale": weight_global_scale.clone(),
                        "act_global_scale": act_global_scale.clone()
                    }

        # 8. Update activations
        device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            out = maybe_first_element(out).to(act_offload_device)
            # change only first input argument
            if len(inp_args) > 0:
                inp_args[0].data = out
            elif "hidden_states" in inp_kwargs:
                inp_kwargs["hidden_states"] = out
            else:
                raise ValueError("Unsupported block input format.")

        if args.cpu_offload_modules:
            block = block.cpu()

        # 8. Clean-up
        del gptq_handles
        del hooks
        clear_device_cache(garbage_collection=True)

    clear_device_cache(garbage_collection=True)

    return quantized_state_dict, non_quantized_state_dict
