# Quantizing Flux Kontext

- Here is a quick example for how to use FP-Quant to quantize the models on the fly.

~~~python

pipe = FluxKontextPipeline.from_pretrained("/home/cropy/flux_kontext", 
                                        local_files_only=True,
                                        quantization_config=pipeline_quant_config,
                                        torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Apply Qutlass quantization to the transformer
# Read the layer analytics (if present) to compare each layer’s quantized runtime with the normal runtime.

try:
    with open("layer_analytics.json", "r") as f:
        layer_analytics_list = json.load(f)
        layer_analytics_list = [key for key in layer_analytics_list if layer_analytics_list[key]["quantized_layer_time"]/layer_analytics_list[key]["nn_layer_time"] > 0.95]
    enable_analytics = False
except:
    layer_analytics_list = []
    print("No layer_analytics.json found, or error in reading it.")
    enable_analytics = True

from fp_quant.inference_lib.src.fp_quant import FPQuantLinear, FPQuantConfig, FPQuantDtype, replace_with_fp_quant_linear
fp_quant_config = FPQuantConfig(forward_dtype=FPQuantDtype.MXFP4, forward_method="abs_max", 
                                backward_dtype=FPQuantDtype.BF16, hadamard_group_size=32,
                                modules_to_not_convert=[
                                    "x_embedder", # we should not quantize x_embedder. Otherwise the resulting image looks like noise.
                                    *layer_analytics_list
                                ],
)

_, result, num_of_params=replace_with_fp_quant_linear(pipe.transformer, fp_quant_config, apply_pre_forward=True, enable_analytics=enable_analytics)
print("Transformer Replaced:", result, "Num of params:", num_of_params)

pipe.transformer = torch.compile(
    pipe.transformer, mode="max-autotune", fullgraph=False
)
pipe.vae.decode = torch.compile(
    pipe.vae.decode, mode="max-autotune", fullgraph=True
)

file_name = "images/1.png"

input_image = load_image(file_name)

height=1024
width=1024
input_image.resize((width, height))
num_images_per_prompt = 1


for _ in range(5):
    images = pipe(
    image=input_image,
    prompt="Your prompt",


    negative_prompt="blurry, low quality, bad quality, worst quality, deformed, distorted, worst quality",
    guidance_scale=2.5,
    height=height, 
    width=width,
    max_area=height*width,
    num_inference_steps=25,
    generator=torch.manual_seed(441),
    num_images_per_prompt=num_images_per_prompt
    ).images


    for i in range(num_images_per_prompt):
        file_name_i = file_name.replace(".jpg", f"_{i}.jpg").replace(".png", f"_{i}.png")
        images[i].save(file_name_i)

~~~
