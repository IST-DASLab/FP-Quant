import os
import json
import argparse
import warnings
from functools import partial

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM

from src.metrics.perplexity import compute_perplexity
from src.transforms.transforms import TRANSFORMS
from src.quantization.quant_ops import NVFP_GROUPSIZE, MXFP_GROUPSIZE
from src.quantization.qconfig import prepare_quantization_config
from src.quantization import rtn_quantization, gptq_quantization
from src.utils.common_utils import fix_seed
from src.utils.data_utils import get_data, get_wikitext2

try:
    import wandb
except ImportError:
    wandb = None

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def auto_or_int(value):
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Must be 'auto' or an integer, got '{value}'")


def export_quantized_model(model, quantized_state_dict, non_quantized_state_dict, args):
    config = model.config
    # Prepare directory to save model
    os.makedirs(args.save_path, exist_ok=True)

    blocks = model.model.layers

    # State dict to save
    model_state_dict = {}

    for block_idx, block in enumerate(blocks):
        prefix = f"model.layers.{block_idx}."
        for k, v in block.state_dict().items():
            layer_name, param_name = k.rsplit(".", 1)
            if f"{prefix}{layer_name}" in quantized_state_dict and param_name == "weight":
                for k_compr, v_compr in quantized_state_dict[f"{prefix}{layer_name}"].items():
                    model_state_dict[f"{prefix}{layer_name}.{k_compr}"] = v_compr.cpu()
            elif f"{prefix}{k}" in non_quantized_state_dict:
                model_state_dict[f"{prefix}{k}"] = non_quantized_state_dict[f"{prefix}{k}"].cpu()
            else:
                model_state_dict[f"{prefix}{k}"] = v.cpu()

    # Add non_quantized_state_dict block parameters (dict is non-empty for blockwise_qat)
    model_state_dict.update(non_quantized_state_dict)

    # Process all remaining blocks
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)

    for k, v in model.state_dict().items():
        if not (k.startswith("model.layers") or (k == "lm_head.weight" and tie_word_embeddings)):
            model_state_dict[k] = v.cpu()

    # Split checkpoint into shards
    current_shard_size = 0
    current_shard = {}
    shards = []

    for k, v in model_state_dict.items():
        tensor_size = v.numel() * v.element_size()
        if current_shard_size + tensor_size > args.max_shard_size:
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        if tensor_size > args.max_shard_size:
            shards.append({k: v})
            continue
        
        current_shard[k] = v
        current_shard_size += tensor_size

    # Dump last shard if it is not empty
    if len(current_shard) > 0:
        shards.append(current_shard)

    safetensors_index = {}
    num_shards = len(shards)
    max_digits = len(str(max(num_shards, 1)))

    # Save shards
    for shard_idx, shard in enumerate(shards):
        current_shard_path = f"model-{str(shard_idx+1).zfill(max_digits)}-of-{str(num_shards).zfill(max_digits)}.safetensors"
        save_file(shard, os.path.join(args.save_path, current_shard_path))
        for k in shard:
            safetensors_index[k] = current_shard_path

    # Save safetensors index
    with open(os.path.join(args.save_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": safetensors_index}, f)

    # Add quantization metadata
    config.quantization_config = prepare_quantization_config(
        args.hadamard_group_size, 
        args.format,
        pseudoquantization=(args.export_quantized_model == "pseudoquant")
    )
    # Save configs
    config.save_pretrained(args.save_path)
    model.generation_config.save_pretrained(args.save_path)

    
def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to quantized model.",
    )
    # Data params
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="The name or path to the calibration dataset.",
    )
    parser.add_argument(
        "--sequence_length", 
        default=2048, 
        type=int, 
        help="Length of calibration sequences."
    )
    parser.add_argument(
        "--num_sequences", 
        default=1024, 
        type=int, 
        help="Number of calibration sequences."
    )
    # Quantization params
    parser.add_argument(
        "--format",
        type=str,
        default="int",
        choices=["int", "fp", "nvfp", "mxfp"],
        help="Quantization format.",
    )
    parser.add_argument(
        "--scale_precision",
        type=str,
        default="fp16",
        choices=["fp16", "e8m0", "e4m3"],
        help="Scale precision.",
    )
    parser.add_argument(
        "--w_granularity",
        type=str,
        default="group",
        choices=["tensor", "channel", "group"],
        help="Weight quantization granularity.",
    )
    parser.add_argument(
        "--w_bits",
        type=int,
        required=True,
        help="Weight quantization bitwidth.",
    )
    parser.add_argument(
        "--w_group_size",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--w_observer",
        type=str,
        default="minmax",
        choices=["minmax", "mse"],
        help="Weight observer.",
    )
    parser.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="Activation quantization bitwidth.",
    )
    parser.add_argument(
        "--a_granularity",
        type=str,
        default="group",
        choices=["tensor", "channel", "group"],
        help="Activation quantization granularity.",
    )
    parser.add_argument(
        "--a_group_size",
        type=int,
        default=None,
        help="How many activation columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--a_observer",
        type=str,
        default="minmax",
        choices=["minmax"],
        help="Activation observer.",
    )
    parser.add_argument(
        "--export_quantized_model",
        type=str,
        default="",
        choices=["", "realquant", "pseudoquant"],
        help="Whether export quantized model in realquant or pseudoquant format.",
    )
    # GPTQ params
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="Run GPTQ quantization.",
    )
    parser.add_argument(
        "--quantization_order",
        type=str,
        default="default",
        choices=["default", "activation"],
        help="Weigth quantization order in GPTQ.",
    )
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    # Transform params
    parser.add_argument(
        "--transform_class",
        type=str,
        default="identity",
        choices=TRANSFORMS.keys(),
        help="The transform class."
    )
    parser.add_argument(
        "--hadamard_group_size",
        type=int,
        default=128,
        help="Hadamard group size"
    )
    # Logging params
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to log to wandb."
    )
    # Misc params
    parser.add_argument(
        "--verbose",
        action="store_true"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed.")
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--amp", action="store_true", help="whether to enable fp16 autocasting.")
    parser.add_argument("--compile", action="store_true", help="whether to use torch.compile.")
    parser.add_argument("--fuse_global_scale", action="store_true", help="whether to fuse global scale in qkv and gate_up.")
    # Eval params
    parser.add_argument("--eval_perplexity", action="store_true", help="whether to eval perplexity after quantization.")
    parser.add_argument("--eval_openllm", action="store_true", help="whether to eval OpenLLM v1 openllm after quantization.")
    # LM eval params
    parser.add_argument(
        "--lm_eval_batch_size",
        type=auto_or_int,
        default="auto",
        help="LM eval batch size to evaluate after quantization.",
    )
    parser.add_argument(
        "--lm_eval_tasks",
        nargs="+",
        type=str,
        default=["mmlu_cot_llama", "arc_challenge_llama", "gsm8k_llama", "hellaswag", "winogrande", "truthfulqa"],
        help="OpenLLMv1 tasks to evaluate after quantization."
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Whether to disable thinking mode for Qwen3.",
    )
    # Save params
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save quantized model",
    )
    parser.add_argument(
        "--max_shard_size", 
        type=int, 
        default=5 * 1024 * 1024 * 1024, 
        help="Maximum shard size in bytes."
    )
    # Parse arguments
    args = parser.parse_args()
    # Check and fix group_size (if needed)
    if args.format == "nvfp":
        if args.w_group_size != NVFP_GROUPSIZE:
            args.w_group_size = NVFP_GROUPSIZE
            print(f"Changed weight group_size to {NVFP_GROUPSIZE} for nvfp format.")
        if args.a_group_size != NVFP_GROUPSIZE:
            args.a_group_size = NVFP_GROUPSIZE
            print(f"Changed activation group_size to {NVFP_GROUPSIZE} for nvfp format.")
        if args.scale_precision != "e4m3":
            args.scale_precision = "e4m3"
            print(f"Changed scale_precision to e4m3 for nvfp format.")
    elif args.format == "mxfp":
        if args.w_group_size != MXFP_GROUPSIZE:
            args.w_group_size = MXFP_GROUPSIZE
            print(f"Changed weight group_size to {MXFP_GROUPSIZE} for mxfp format.")
        if args.a_group_size != MXFP_GROUPSIZE:
            args.a_group_size = MXFP_GROUPSIZE
            print(f"Changed activation group_size to {MXFP_GROUPSIZE} for mxfp format.")
        if args.scale_precision != "e8m0":
            args.scale_precision = "e8m0"
            print(f"Changed scale precision to e8m0 for mxfp format.")
    # Check logging
    if args.log_wandb:
        assert wandb is not None, "wandb is not installed. Please install wandb `pip install wandb`."
    # Check real_quant config
    if args.export_quantized_model:
        assert args.save_path is not None, "`save_path` must be specified when exporting quantized model."
        assert args.format in ["nvfp", "mxfp"], "`export_quantization` is only supported for nvfp and mxfp formats."
        assert args.w_bits == 4, "`export_quantization` is only supported for 4 bit weights."
        assert args.a_bits == 4, "`export_quantization` is only supported for 4 bit activations."
    return args


def main():
    args = parse_args()
    # Fix seed
    fix_seed(args.seed)
    # Set device
    device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    # Get dtype
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Init logger
    if args.log_wandb:
        wandb.init(config=args)
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=args.dtype, 
        device_map=None if args.cpu_offload_modules else device,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Sanity check
    if args.eval_openllm:
        assert hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None, "OpenLLM v1 works only with chat template."
        if args.disable_thinking:
            if model.config.model_type == "qwen3":
                tokenizer.apply_chat_template = partial(
                    tokenizer.apply_chat_template, 
                    enable_thinking=False
                )
            else:
                warnings.warn("`disable_thinking` has no effect on non-Qwen3 models.")

    quantize_anything = args.w_bits < 16 or args.a_bits < 16

    # Prepare calibration data
    calibration_data = get_data(
        args.dataset_name_or_path,
        tokenizer,
        args.sequence_length,
        args.num_sequences,
        args.seed
    )

    if quantize_anything:
        if args.gptq:
            quantized_state_dict, non_quantized_state_dict = gptq_quantization(model, calibration_data, args, device)
        else:
            quantized_state_dict, non_quantized_state_dict = rtn_quantization(model, calibration_data, args, device)

        if args.export_quantized_model:
            export_quantized_model(model, quantized_state_dict, non_quantized_state_dict, args) 
            tokenizer.save_pretrained(args.save_path)

    if args.compile:
        model = torch.compile(model)

    if args.eval_perplexity or args.eval_openllm:
        model = model.to(device)

    if args.eval_perplexity:
        eval_data = get_wikitext2(tokenizer, args.sequence_length)
        ppl = compute_perplexity(model, eval_data)
        print(f"Wikitext-2 perplexity: {round(ppl, 2):.2f}")
        if args.log_wandb:
            wandb.log({"eval/wikitext2_ppl": ppl})

    # OpenLLM v1 openllm (following https://arxiv.org/abs/2411.02355)
    if args.eval_openllm:

        results = {}
        lm = HFLM(
            pretrained=model, 
            tokenizer=tokenizer, 
            batch_size=args.lm_eval_batch_size,
            max_length=4096, # from open LLM openllm
        )
        task_manager = lm_eval.tasks.TaskManager()

        # Winogrande (5-shot)
        if "winogrande" in args.lm_eval_tasks:
            task_results = lm_eval.simple_evaluate(
                model=lm,
                tasks="winogrande",
                num_fewshot=5,
                batch_size=args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"]
            results.update(task_results)
            print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
        # Hellaswag (10-shot)
        if "hellaswag" in args.lm_eval_tasks:
            task_results = lm_eval.simple_evaluate(
                model=lm,
                tasks="hellaswag",
                num_fewshot=10,
                batch_size=args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"]
            results.update(task_results)
            print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
        # GSM8K Llama-3.1
        if "gsm8k_llama" in args.lm_eval_tasks:
            task_results = lm_eval.simple_evaluate(
                model=lm,
                tasks="gsm8k_llama",
                batch_size=args.lm_eval_batch_size,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                task_manager=task_manager,
            )["results"]
            results.update(task_results)
            print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_better": {}}))
        # MMLU CoT Llama-3.1
        if "mmlu_cot_llama" in args.lm_eval_tasks:
            task_results = lm_eval.simple_evaluate(
                model=lm,
                tasks="mmlu_cot_llama",
                batch_size=args.lm_eval_batch_size,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                task_manager=task_manager,
            )["results"]
            results.update(task_results)
            print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_better": {}}))
        # Log results
        if args.log_wandb:
            wandb.log({"eval/openllm": results}) 
        # Print formatted table
        print("### Final results ###")
        print(make_table({"results": results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))


if __name__ == "__main__":
    main()
