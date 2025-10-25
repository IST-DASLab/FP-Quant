# Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization 

<a href='https://arxiv.org/pdf/2509.23202'><img src='https://img.shields.io/badge/ArXiv-PDF-red' height="25"></a> &nbsp; 

The official implementation for the paper [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202).

This repository contains the code needed to reproduce the results presented in the paper, and it also offers the ability to export quantized models with [QuTLASS](https://github.com/IST-DASLab/qutlass) kernels in the **MXFP4** and **NVFP4 formats**. The exported models can be run either with Hugging Face Transformers or with vLLM.

### Repository structure
---

The repository is structured as follows:

* `model_quant.py` - the main quantization script
* `src/` - source code with implementation of all necessary functionality \
    ```├── quantization``` - quantization functionality \
    ```├── transforms``` - transform functionality \
    ```├── utils``` - utility functions

### Environment setup
---

**Inference Engines**

FP-Quant has support implemented in:
 - `transformers` with these features:
     - Available in `main` ([Documentation](https://huggingface.co/docs/transformers/main/en/quantization/fp_quant#fp-quant)).
     - RTN on-the-fly quantization.
       ```python
       from transformers import AutoModelForCausalLM, AutoTokenizer, FPQuantConfig
       import torch
        
       model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            quantization_config=FPQuantConfig(forward_dtype="mxfp4"),
            device_map="auto",
            dtype=torch.bfloat16,
        )
       model.forward = torch.compile(model.forward, mode="max-autotune", fullgraph=True)
       ```
     - Pseudo-quantization QAT.
 - `vLLM` with these features:
     - Available in [this PR](https://github.com/vllm-project/vllm/pull/24440).
     - Compatible with real quantization models from `FP-Quant` and the `transformers` integration.

### FP-Quant models
---

👉 Check out the quantized MXFP and NVFP models in the [MR-GPTQ](https://huggingface.co/collections/ISTA-DASLab/mr-gptq-68dcde4b1e4b572ded89dbf3) collection on Hugging Face 🤗.  

*Example of quantized model inference with HF*
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, FPQuantConfig
import torch

model_name = "ISTA-DASLab/Llama-3.1-8B-Instruct-MR-GPTQ-nvfp"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
)
prompt = "Explain quantization for neural network in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    output_tokens = model.generate(**inputs,max_new_tokens=150 )
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(generated_text)
```  
*Example of quantized model inference with vLLM engine*  

```python
from vllm import LLM, SamplingParams

model_name = "ISTA-DASLab/Llama-3.1-8B-Instruct-MR-GPTQ-nvfp"
llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.9)
sampling_params = SamplingParams(
    temperature=0.7,       # creativity
    top_p=0.9,             # nucleus sampling
    max_tokens=150,        # number of new tokens to generate
)
prompt = "Explain quantization for neural networks in simple terms."
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```
### Quantization
---

**NOTE** - The quantization script is designed to be run on a single GPU.

**NOTE** - Only Llama and Qwen3 models are supported.

Below is an example of the quantization script usage:

```shell
#!/bin/bash
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

MODEL=${MODEL:-"meta-llama/Llama-3.2-1B-Instruct"}
MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
# Data params
NUM_SEQUENCES=${NUM_SEQUENCES:-128}
# Quantization params
FORMAT=${FORMAT:-"nvfp"}
W_BITS=${W_BITS:-4}
A_BITS=${A_BITS:-16}
W_GROUP_SIZE=${W_GROUP_SIZE:-16}
A_GROUP_SIZE=${A_GROUP_SIZE:-16}
GPTQ=${GPTQ:-0}
W_OBSERVER=${W_OBSERVER:-"minmax"}
QUANTIZATION_ORDER=${QUANTIZATION_ORDER:-"default"}
# Save params
EXPORT_QUANTIZATION=${EXPORT_QUANTIZATION:-""}
# Transform params
TRANSFORM_CLASS=${TRANSFORM_CLASS:-"identity"}
HADAMARD_GROUP_SIZE=${HADAMARD_GROUP_SIZE:-128}
# Evaluation params
EVAL_PERPLEXITY=${EVAL_PERPLEXITY:-1}
EVAL_OPENLLM=${EVAL_OPENLLM:-0}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-"auto"}
# Misc params
LOG_WANDB=${LOG_WANDB:-0}
DTYPE=${DTYPE:-"auto"}
CPU_OFFLOAD_ACTIVATIONS=${CPU_OFFLOAD_ACTIVATIONS:-0}

SCRIPT_ARGS=""

if [[ $GPTQ == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --gptq"
fi

if [[ $EVAL_PERPLEXITY == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_perplexity"
fi

if [[ $EVAL_OPENLLM == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_openllm"
fi

if [[ $LOG_WANDB == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --log_wandb"
fi

METHOD_NAME=""
if [[ $GPTQ == 1 ]]; then
    METHOD_NAME="GPTQ"
else
    METHOD_NAME="RTN"
fi

if [[ $CPU_OFFLOAD_MODULES == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --cpu_offload_modules"
fi

if [[ $CPU_OFFLOAD_ACTIVATIONS == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --cpu_offload_activations"
fi

export WANDB_PROJECT="FP-Quantization-Harness"
export WANDB_NAME=${MODEL}/${FORMAT}-w${W_BITS}-a${A_BITS}-${METHOD_NAME}-${TRANSFORM_CLASS}-transform

if [[ $EXPORT_QUANTIZATION == "realquant" || $EXPORT_QUANTIZATION == "pseudoquant" ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --export_quantized_model ${EXPORT_QUANTIZATION}"
    if [[ $EXPORT_QUANTIZATION == "realquant" ]]; then
        SAVE_DIR=quantized_models
    else
        SAVE_DIR=pseudoquantized_models
    fi
fi

python model_quant.py \
    --model_name_or_path=${MODEL} \
    --format=${FORMAT} \
    --w_bits=${W_BITS} \
    --a_bits=${A_BITS} \
    --w_group_size=${W_GROUP_SIZE} \
    --a_group_size=${A_GROUP_SIZE} \
    --transform_class=${TRANSFORM_CLASS} \
    --w_observer=${W_OBSERVER} \
    --quantization_order=${QUANTIZATION_ORDER} \
    $SCRIPT_ARGS \
    --hadamard_group_size=${HADAMARD_GROUP_SIZE} \
    --dataset_name_or_path=fineweb-edu \
    --num_sequences=${NUM_SEQUENCES} \
    --sequence_length=2048 \
    --dtype=${DTYPE} \
    --lm_eval_batch_size=${LM_EVAL_BATCH_SIZE} \
    --save_path "${SAVE_DIR}/${MODEL_ID}-${FORMAT}-w${W_BITS}-a${A_BITS}-${METHOD_NAME}-${TRANSFORM_CLASS}-transform" \
    --export_quantized_model pseudoquant \
    --cpu_offload_activations \
    --cpu_offload_modules \
    --fuse_global_scale \
    --amp
```

Above:
* `--model_name_or_path` - The model to quantize. (Llama and Qwen3 models are supported)
* `--format` - The quantization format (int, fp, mxfp, nvfp). 
* `--w_bits` - The number of bits to quantize the weights to.
* `--a_bits` - The number of bits to quantize the activations to.
* `--w_group_size` - The number of weights to quantize together.
* `--a_group_size` - The number of activations to quantize together.
* `--init` - Transform initialization.
* `--transform_class` - Transform class. We provide the following options:
    * `identity` - Identity transform
    * `hadamard` - Hadamard transform
    * `dct` - Discrete cosine transform
    * `dst` - Discrete sine transform
    * `fast_food` - Fast food transform
    * `gsr` - Grouped sequency aligned transform
* `--hadamard_group_size` - Transform group size.
* `--dataset_name_or_path` - Dataset to use for calibration.
* `--sequence_length` - Calibration sequence length.
* `--dtype` - Data type to load the model.
* `--amp` - Whether to use automatic mixed precision.
* `--export_quantized_model` - Whether to export quantized model in `realquant` or `pseudoquant` format. The former allows one to run quantized model with the help of [QuTLASS](https://github.com/IST-DASLab/qutlass) integration, while the latter produces fake quantized model runnable with `triton` kernels.

For evaluation, we provide the following options:

* `--eval_perplexity` - Whether to evaluate perplexity after quantization.
* `--eval_openllm` - Whether to evaluate OpenLLM v1 openllm after quantization.
* `--lm_eval_batch_size` - LM eval batch size to evaluate after quantization.
* `--fuse_global_scale` - Whether to fuse global scale in qkv and gate_up projections as required by `vLLM`.


We note, however, that the evaluation within quantization script is not optimized and may take several days.
The recommended way to evaluate models is to export the quantized model and evaluate it via `vLLM` integration.

*Evaluation*

We evaluate the compressed models on a subset of the tasks from OpenLLM v1 benchmark using the recommended parameters.

Below is an example of the bash evaluation script usage:

```shell
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn

NUM_GPUS=$( echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l )
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
MODEL_ARGS="pretrained=$MODEL,max_model_len=4096,tensor_parallel_size=$NUM_GPUS,dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},enforce_eager=True"

# Winogrande
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks winogrande \
  --num_fewshot=5 \
  --batch_size auto \
  --output_path lm_eval_results

# Hellaswag
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks hellaswag \
  --num_fewshot=10 \
  --batch_size auto \
  --output_path lm_eval_results

# GSM8k
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks gsm8k_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto \
  --output_path lm_eval_results

# MMLU-CoT 
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks mmlu_cot_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto \
  --output_path lm_eval_results
```


### Citation
---

If you find this project useful, please cite our paper:

```
@misc{egiazarian2025bridginggappromiseperformance,
      title={Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization}, 
      author={Vage Egiazarian and Roberto L. Castro and Denis Kuznedelev and Andrei Panferov and Eldar Kurtic and Shubhra Pandit and Alexandre Marques and Mark Kurtz and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
      year={2025},
      eprint={2509.23202},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.23202}, 
}
```
