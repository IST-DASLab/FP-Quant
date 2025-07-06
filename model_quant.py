import os
import argparse

import torch
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

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def auto_or_int(value):
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Must be 'auto' or an integer, got '{value}'")


def save_quantized_model(model, quantized_state_dict, args):
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
            else:
                model_state_dict[f"{prefix}{k}"] = v.cpu()

    # Save all remaining blocks
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)
    for k, v in model.state_dict().items():
        if not (k.startswith("model.layers") or (k == "lm_head.weight" and tie_word_embeddings)):
            model_state_dict[k] = v.cpu()

    torch.save(model_state_dict, os.path.join(args.save_path, "pytorch_model.bin"))

    # Add quantization metadata
    config.quantization_config = prepare_quantization_config(args.w_group_size, args.format)
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
        default=128, 
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
        "--real_quant",
        action="store_true",
        help="Whether to apply real quantization to model."
    )
    parser.add_argument(
        "--mxfp_scale_factor",
        type=float,
        default=1.0,
        help="MXFP scale scaling factor for MXFP quantization."
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
        default=["arc_easy", "arc_challenge", "winogrande", "piqa", "hellaswag"],
        help="LM eval tasks to evaluate after quantization.",
    )
    parser.add_argument(
        "--lm_eval_add_bos_token", 
        action="store_true",
        help="whether to add bos token in evaluation."
    )
    parser.add_argument(
        "--lm_eval_apply_chat_template",
        action="store_true",
        help="whether to apply chat template."
    )
    parser.add_argument(
        "--lm_eval_fewshot_as_multiturn",
        action="store_true",
        help="whether to process fewshot as multiturn." 
    )
    # Save params
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save quantized model",
    )
    parser.add_argument(
        "--blocks_per_shard", 
        type=int, 
        default=8, 
        help="Number of blocks per shard."
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
    # Check equality of w_group_size and a_group_size
    assert args.w_group_size == args.a_group_size, "w_group_size and a_group_size must be equal."
    # Check logging
    if args.log_wandb:
        assert wandb is not None, "wandb is not installed. Please install wandb `pip install wandb`."
    # Check real_quant config
    if args.real_quant:
        assert args.a_bits == 16, "Real quantization is only supported for weight-only quantization."
        assert args.format in "mxfp", "Real quantization is only supported for mxfp format."
    
    return args


def main():
    args = parse_args()
    # Fix seed
    fix_seed(args.seed)
    # Set device
    device = "cuda"
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

    quantize_anything = args.w_bits < 16 or args.a_bits < 16

    if quantize_anything:
        if args.gptq:
            calibration_data = get_data(
                args.dataset_name_or_path,
                tokenizer,
                args.sequence_length,
                args.num_sequences,
                args.seed
            )
            quantized_state_dict = gptq_quantization(model, calibration_data, args, device=device)
        else:
            quantized_state_dict = rtn_quantization(model, args, device)

        if args.save_path:
            # Custom saving function
            if args.real_quant:
                save_quantized_model(model, quantized_state_dict, args)
            else:
                model.save_pretrained(args.save_path)  
            tokenizer.save_pretrained(args.save_path)
            
    # Set to eval mode
    model.requires_grad_(False).eval()

    if args.compile:
        model = torch.compile(model)

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
            add_bos_token=args.lm_eval_add_bos_token
        )
        task_manager = lm_eval.tasks.TaskManager()

        # MMLU CoT Llama-3.1
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="mmlu_cot_llama",
                batch_size=args.lm_eval_batch_size,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                task_manager=task_manager,
            )["results"]
        )
        # ArcC Llama-3.1
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="arc_challenge_llama",
                batch_size=args.lm_eval_batch_size,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                task_manager=task_manager,
            )["results"]
        )
        # GSM8K Llama-3.1
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="gsm8k_llama",
                batch_size=args.lm_eval_batch_size,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                task_manager=task_manager,
            )["results"]
        )
        # Hellaswag (10-shot)
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="hellaswag",
                num_fewshot=10,
                batch_size=args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"]
        )
        # Winogrande (5-shot)
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="winogrande",
                num_fewshot=5,
                batch_size=args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"]
        )
        # TruthfulQA (0-shot)
        results.update(
            lm_eval.simple_evaluate(
                model=lm,
                tasks="truthfulqa",
                num_fewshot=0,
                batch_size=args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"]
        )

        # Log results
        if args.log_wandb:
            wandb.log({"eval/openllm": results}) 
        # Print formatted table
        print(make_table({"results": results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))


if __name__ == "__main__":
    main()
