import argparse
import os
import runpy
from functools import partial
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_prompt_list(path: Path, variable_name: str) -> list[str]:
    prompts = runpy.run_path(str(path)).get(variable_name)
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError(f"prompts file must define {variable_name} as a list[str]")
    if not prompts:
        raise ValueError(f"{variable_name} is empty")
    return prompts


def find_layers_module(model):
    candidates = [
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "transformer.h",
        "layers",
    ]
    for path in candidates:
        curr = model
        try:
            for part in path.split("."):
                curr = getattr(curr, part)
        except AttributeError:
            continue
        if hasattr(curr, "__len__") and len(curr) > 0:
            print(f"Found transformer layers at '{path}' ({len(curr)} layers)")
            return curr
    raise AttributeError("Could not find transformer layers on this model")


def get_down_proj(layer):
    candidates = [
        "mlp.down_proj",
        "feed_forward.w2",
        "ffn.down_proj",
    ]
    for path in candidates:
        curr = layer
        try:
            for part in path.split("."):
                curr = getattr(curr, part)
        except AttributeError:
            continue
        if hasattr(curr, "weight"):
            return curr
    raise AttributeError("Could not find an MLP down projection weight on this layer")


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hook_fn(module, inputs, output, layer_idx, captured):
    hidden = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(hidden):
        return
    if hidden.dim() == 3:
        token_act = hidden[:, -1, :]
    elif hidden.dim() == 2:
        token_act = hidden[-1, :].unsqueeze(0)
    else:
        return
    captured[layer_idx] = token_act.detach().cpu().float()


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt


def get_mean_activations(model, tokenizer, layers_module, target_layers, prompts, desc, device):
    sums = {layer_idx: None for layer_idx in target_layers}
    counts = {layer_idx: 0 for layer_idx in target_layers}

    for prompt in tqdm(prompts, desc=desc):
        text = format_prompt(tokenizer, prompt)
        inputs = tokenizer([text], return_tensors="pt").to(device)

        captured = {}
        hooks = [
            layers_module[layer_idx].register_forward_hook(
                partial(hook_fn, layer_idx=layer_idx, captured=captured)
            )
            for layer_idx in target_layers
        ]

        with torch.inference_mode():
            model(**inputs)

        for hook in hooks:
            hook.remove()

        for layer_idx, act in captured.items():
            sums[layer_idx] = act if sums[layer_idx] is None else sums[layer_idx] + act
            counts[layer_idx] += 1

    means = {}
    for layer_idx in target_layers:
        means[layer_idx] = None if sums[layer_idx] is None else sums[layer_idx] / counts[layer_idx]
    return means


def generate_once(model, tokenizer, prompt: str, max_new_tokens: int, device):
    text = format_prompt(tokenizer, prompt)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def assert_transformers_model_path(model_path: Path):
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        raise ValueError(
            "GGUF/LM Studio files do not expose PyTorch layers or editable weights. "
            "Use a local Hugging Face/Transformers model directory with config.json and model weights instead."
        )
    if model_path.is_dir() and not (model_path / "config.json").exists():
        raise ValueError(
            f"{model_path} does not look like a Transformers model directory (missing config.json)."
        )


def main():
    parser = argparse.ArgumentParser(description="Run a local activation ablation smoke test on HF models.")
    parser.add_argument(
        "--model",
        default=str(SCRIPT_DIR / "Qwen3-1.7B"),
        help="Local Transformers model directory.",
    )
    parser.add_argument("--layers", default="8-18", help="Layer list, for example '2,3' or '10-25'.")
    parser.add_argument("--output", default="surgery-output", help="Where to save the edited model.")
    parser.add_argument("--ablation-scale", type=float, default=1)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda.")
    parser.add_argument("--skip-save", action="store_true", help="Do not save the edited model.")
    parser.add_argument("--test-prompt", default="Write a short Python function that adds two numbers.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--load-cpu",
        action="store_true",
        help="Force load all weights on CPU (slower but avoids OOM when model > GPU VRAM)."
    )
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    assert_transformers_model_path(model_path)

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    harmful_prompts = load_prompt_list(SCRIPT_DIR / "harmful_prompts.py", "harmful_prompts")
    harmless_prompts = load_prompt_list(SCRIPT_DIR / "harmless_prompts.py", "harmless_prompts")

    device = pick_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading local model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )

    # ── CPU/GPU hybrid loading via device_map="auto" (for models > GPU VRAM) ──
    from accelerate import infer_auto_device_map
    import psutil

    ram_gb = psutil.virtual_memory().total / (1024**3)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    gpu_mem_gb = gpu_mem / (1024**3)

    # Estimate model FP16 size from safetensors files
    model_size_gb = sum(
        (model_path / f).stat().st_size
        for f in os.listdir(model_path)
        if f.endswith('.safetensors')
    ) / (1024**3)
    print(f"System RAM: {ram_gb:.0f}GB, GPU VRAM: {gpu_mem_gb:.1f}GB, Model size: {model_size_gb:.1f}GB")

    if args.load_cpu or (torch.cuda.is_available() and model_size_gb + 2 > gpu_mem_gb):
        # Model too large for GPU (+2GB overhead) — use CPU
        print(f"Model ({model_size_gb:.1f}GB) + overhead exceeds GPU ({gpu_mem_gb:.1f}GB), loading on CPU.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype_map[args.dtype],
            trust_remote_code=True,
            local_files_only=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    elif torch.cuda.is_available():
        # Model fits in GPU — use CUDA
        print(f"Model fits in GPU, loading on CUDA.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype_map[args.dtype],
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype_map[args.dtype],
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)

    # Show device distribution
    if hasattr(model, "hf_device_map"):
        devices_used = set(str(d) for d in model.hf_device_map.values())
        print(f"Devices used: {devices_used}")
    model.eval()

    # Determine input device for model with device_map
    input_device = device  # default
    if hasattr(model, "hf_device_map"):
        # Check if any layer is on cuda
        any_cuda = any(
            str(d) != "cpu" and ("cuda" in str(d) or str(d).isdigit())
            for d in model.hf_device_map.values()
        )
        input_device = torch.device("cuda") if any_cuda else torch.device("cpu")
    print(f"Input device: {input_device}")

    layers_module = find_layers_module(model)
    target_layers = parse_layers(args.layers)
    max_layer = len(layers_module) - 1
    invalid_layers = [idx for idx in target_layers if idx < 0 or idx > max_layer]
    if invalid_layers:
        raise ValueError(f"Invalid layer indices {invalid_layers}; model has layers 0..{max_layer}")

    print("\nPre-surgery test:")
    print(generate_once(model, tokenizer, args.test_prompt, args.max_new_tokens, input_device))

    print("\nCollecting activations...")
    harmful_acts = get_mean_activations(
        model, tokenizer, layers_module, target_layers, harmful_prompts, "harmful", input_device
    )
    harmless_acts = get_mean_activations(
        model, tokenizer, layers_module, target_layers, harmless_prompts, "harmless", input_device
    )

    print(f"\nApplying ablation, scale={args.ablation_scale}")
    edited_layers = 0
    for layer_idx in tqdm(target_layers, desc="ablating"):
        harmful = harmful_acts[layer_idx]
        harmless = harmless_acts[layer_idx]
        if harmful is None or harmless is None:
            print(f"Skipping layer {layer_idx}: activation was not captured")
            continue

        refusal_vec = harmful - harmless
        norm = torch.norm(refusal_vec)
        if norm == 0:
            print(f"Skipping layer {layer_idx}: zero activation difference")
            continue

        refusal_vec = refusal_vec / norm
        down_proj = get_down_proj(layers_module[layer_idx])
        weight = down_proj.weight.data
        if not torch.is_floating_point(weight):
            raise TypeError(f"Layer {layer_idx} weight is not floating point; quantized weights are not editable here")

        v = refusal_vec.to(device=weight.device, dtype=torch.float32).T
        weight_fp32 = weight.float()
        projected = weight_fp32 - args.ablation_scale * torch.mm(v, torch.mm(v.T, weight_fp32))
        down_proj.weight.data.copy_(projected.to(dtype=weight.dtype))
        edited_layers += 1

    print(f"Edited {edited_layers} layers")

    print("\nPost-surgery test:")
    print(generate_once(model, tokenizer, args.test_prompt, args.max_new_tokens, input_device))

    if args.skip_save:
        print("\nSkipping save because --skip-save was set.")
        return

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving edited model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done")


if __name__ == "__main__":
    main()
