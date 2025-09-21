import torch
import json

from ..utils.config import FPQuantConfig, FPQuantDtype


all_added_layer_names = []

layer_list = {
    
}

def bench_ms(fn, warmup=10, iters=100):
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start_evt.record()
        _ = fn()
        end_evt.record()
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))  # ms
    t = torch.tensor(times, device="cpu")
    return float(t.mean().item()), float(t.std(unbiased=False).item())

def add_layer(layer: torch.nn.Module, layer_name: str, input_shape: torch.Size, in_features: int, out_features: int, device, dtype):
    
    global layer_list

    if layer_name in layer_list:
        return
    
    layer_info = {
        "config": layer.config,
        "bias": layer.bias is not None,
        "layer_name": layer_name,
        "input_shape": list(input_shape),
        "in_features": in_features,
        "out_features": out_features,
        "device": str(device),
        "dtype": str(dtype),
    }

    layer_list[layer_name] = layer_info
    
    print(f"Layer name: {layer_name}, input shape: {input_shape}, in_features: {in_features}, out_features: {out_features}")

    if len(all_added_layer_names) == len(layer_list):
        analyze_layers()
        print("All layers have been analyzed.")
        
def analyze_layers():

    # First find all unique pairs of (input_shape, in_features, out_features, device, dtype)

    global layer_list

    unique_layers = {}

    for name in layer_list:
        bias = layer_list[name]["bias"]
        input_shape = tuple(layer_list[name]["input_shape"])
        in_features = layer_list[name]["in_features"]
        out_features = layer_list[name]["out_features"]
        device = layer_list[name]["device"]
        dtype = layer_list[name]["dtype"]
        config = layer_list[name]["config"]

        key = (input_shape, in_features, out_features, device, dtype, bias)

        # Only keep the first layer encountered for each unique key
        if key not in unique_layers:
            unique_layers[key] = {
                "config": config
            }

    # Now for each unique layer
    for key in unique_layers:
        input_shape, in_features, out_features, device, dtype, bias = key
        config = unique_layers[key]["config"]

        device = torch.device(device)
        dtype = torch.__dict__[dtype.split('.')[-1]]
        nn_layer = torch.nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        from .linear import FPQuantLinear
        quantized_layer = FPQuantLinear(in_features, out_features, bias=bias, config=config, device=device, dtype=dtype)
        quantized_layer.pre_forward()

        with torch.no_grad():
            sample_input = torch.randn(*input_shape, device=device, dtype=dtype)

            def b1():
                return nn_layer(sample_input)
            
            def b2():
                return quantized_layer(sample_input)

            if input_shape[0] == 12288 or input_shape[1] == 12288:
                print(f"AAAAAAAAAAAAAAAA")

            quantized_layer_time, _ = bench_ms(b2, warmup=100, iters=200)
            nn_layer_time, _ = bench_ms(b1, warmup=100, iters=200)

            if input_shape[0] == 12288 or input_shape[1] == 12288:
                print(f"AAAAAAAAAAAAAAAA")
            
            print(f"Layer: {key}, nn_layer_time: {nn_layer_time:.3f} ms, quantized_layer_time: {quantized_layer_time:.3f} ms, ratio: {quantized_layer_time / nn_layer_time:.3f}")

            unique_layers[key]["quantized_layer_time"] = quantized_layer_time
            unique_layers[key]["nn_layer_time"] = nn_layer_time
        
        del quantized_layer
        del nn_layer
        torch.cuda.empty_cache()

    for name in layer_list:
        input_shape = tuple(layer_list[name]["input_shape"])
        in_features = layer_list[name]["in_features"]
        out_features = layer_list[name]["out_features"]
        device = layer_list[name]["device"]
        dtype = layer_list[name]["dtype"]
        key = (input_shape, in_features, out_features, device, dtype, bias)
        layer_list[name]["quantized_layer_time"] = unique_layers[key]["quantized_layer_time"]
        layer_list[name]["nn_layer_time"] = unique_layers[key]["nn_layer_time"]
        del layer_list[name]["config"]

    # Save to file
    with open("layer_analytics.json", "w") as f:
        json.dump(layer_list, f, indent=4)

