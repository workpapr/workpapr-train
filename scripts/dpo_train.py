#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) training using Apple MLX.

Takes SFT-trained LoRA adapters as starting point and further aligns
the model using preference pairs (chosen vs rejected responses).

Usage:
    python scripts/dpo_train.py \
        --model mlx-community/Llama-3.2-3B-Instruct-4bit \
        --adapter-path .workpapr/demo/mlx-training/sft-adapters \
        --data .workpapr/demo/mlx-training/dpo_pairs.jsonl \
        --output-adapter-path .workpapr/demo/mlx-training/dpo-adapters \
        --iters 50 --learning-rate 5e-6 --beta 0.1
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.utils import load as load_model
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters


def load_dpo_data(data_path: str) -> list[dict]:
    """Load preference pairs from JSONL file."""
    pairs = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def get_log_probs(model, tokenizer, prompt: str, response: str) -> mx.array:
    """Compute sum of per-token log probabilities of response given prompt."""
    prompt_tokens = tokenizer.encode(prompt)
    response_tokens = tokenizer.encode(response)

    if not response_tokens:
        return mx.array(0.0)

    # Full sequence: prompt + response
    full_tokens = prompt_tokens + response_tokens
    # Truncate to avoid OOM on long sequences
    max_len = 512
    if len(full_tokens) > max_len:
        # Keep prompt start + response end
        prompt_tokens = prompt_tokens[: max_len - len(response_tokens)]
        full_tokens = prompt_tokens + response_tokens

    input_ids = mx.array(full_tokens[:-1])[None]  # [1, seq_len-1]
    targets = mx.array(full_tokens[1:])[None]  # [1, seq_len-1]

    logits = model(input_ids)  # [1, seq_len-1, vocab]

    # Only score response portion (not prompt)
    response_start = max(0, len(prompt_tokens) - 1)
    response_logits = logits[:, response_start:, :]
    response_targets = targets[:, response_start:]

    # Numerically stable log softmax
    log_probs = response_logits - mx.logsumexp(response_logits, axis=-1, keepdims=True)

    # Gather log probs for actual next tokens
    gathered = mx.take_along_axis(
        log_probs, response_targets[:, :, None], axis=-1
    ).squeeze(-1)

    return gathered.sum()


def train(args):
    """Run DPO training loop."""
    print(f"Loading base model: {args.model}", flush=True)
    policy_model, tokenizer = load_model(args.model)

    # Load SFT adapters if provided (load_adapters applies LoRA + loads weights in one call)
    if args.adapter_path and Path(args.adapter_path).exists():
        print(f"Loading SFT adapters from: {args.adapter_path}", flush=True)
        policy_model = load_adapters(policy_model, args.adapter_path)
    else:
        print("No SFT adapters found, starting from base model", flush=True)
        lora_config = {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 2.0}
        linear_to_lora_layers(policy_model, args.lora_layers, lora_config)

    # Load reference model (frozen, no LoRA)
    print("Loading reference model (frozen)...", flush=True)
    ref_model, _ = load_model(args.model)
    ref_model.eval()
    ref_model.freeze()

    # Load training data
    print(f"Loading DPO pairs from: {args.data}", flush=True)
    pairs = load_dpo_data(args.data)
    if not pairs:
        print("Error: No training pairs found", flush=True)
        sys.exit(1)
    print(f"Loaded {len(pairs)} preference pairs", flush=True)

    # SGD is more stable than Adam for DPO on quantized models
    optimizer = optim.SGD(learning_rate=args.learning_rate)

    # Training loop — one pair per iteration (cycle through data)
    print(
        f"\nStarting DPO training: {args.iters} iterations, beta={args.beta}, lr={args.learning_rate}",
        flush=True,
    )

    peak_mem = 0.0

    # Precompute reference log probs (constant — no gradients needed)
    print("Precomputing reference log probs...", flush=True)
    ref_cache = []
    for pair in pairs:
        rc = get_log_probs(ref_model, tokenizer, pair["prompt"], pair["chosen"])
        rr = get_log_probs(ref_model, tokenizer, pair["prompt"], pair["rejected"])
        mx.eval(rc, rr)
        ref_cache.append((rc.item(), rr.item()))
    print(f"Reference log probs computed for {len(pairs)} pairs", flush=True)

    # Free reference model memory
    del ref_model

    for iteration in range(1, args.iters + 1):
        iter_start = time.time()
        idx = (iteration - 1) % len(pairs)
        pair = pairs[idx]
        ref_c, ref_r = ref_cache[idx]

        def loss_fn(model):
            # Policy log probs
            pi_chosen = get_log_probs(model, tokenizer, pair["prompt"], pair["chosen"])
            pi_rejected = get_log_probs(
                model, tokenizer, pair["prompt"], pair["rejected"]
            )

            # DPO: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
            log_ratio_chosen = pi_chosen - ref_c
            log_ratio_rejected = pi_rejected - ref_r
            logit = args.beta * (log_ratio_chosen - log_ratio_rejected)
            # Numerically stable: log_sigmoid(x) = -softplus(-x)
            loss = nn.losses.binary_cross_entropy(
                mx.sigmoid(logit), mx.array(1.0), reduction="none"
            )
            return loss

        loss, grads = nn.value_and_grad(policy_model, loss_fn)(policy_model)

        # Check for NaN grads before applying
        grad_flat = tree_flatten(grads)
        grad_norms = [mx.sqrt((v * v).sum()).item() for k, v in grad_flat if v.size > 0]
        max_grad = max(grad_norms) if grad_norms else 0.0
        has_nan_grad = any(math.isnan(g) or math.isinf(g) for g in grad_norms)

        if not has_nan_grad:
            optimizer.update(policy_model, grads)

        mx.eval(policy_model.parameters(), optimizer.state)

        loss_val = loss.item()
        iter_time = time.time() - iter_start
        it_per_sec = 1.0 / iter_time if iter_time > 0 else 0

        try:
            peak_mem = max(peak_mem, mx.get_peak_memory() / 1e9)
        except AttributeError:
            pass

        # Print in mlx_lm.lora-compatible format for consistent parsing
        print(
            f"Iter {iteration}: Train loss {loss_val:.3f}, "
            f"Learning Rate {args.learning_rate:.3e}, "
            f"It/sec {it_per_sec:.2f}, "
            f"Peak mem {peak_mem:.2f} GB",
            flush=True,
        )

    print(f"\nDPO training complete", flush=True)

    # Save adapters in safetensors format
    output_path = Path(args.output_adapter_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect LoRA weights only (matching SFT adapter format)
    all_params = tree_flatten(policy_model.trainable_parameters())
    lora_weights = {k: v for k, v in all_params if "lora" in k}
    print(f"Saving {len(lora_weights)} LoRA weight tensors", flush=True)
    mx.save_safetensors(str(output_path / "adapters.safetensors"), lora_weights)

    # Save adapter config matching mlx_lm format
    # Read the SFT adapter config to preserve LoRA settings
    sft_config_path = Path(args.adapter_path) / "adapter_config.json" if args.adapter_path else None
    if sft_config_path and sft_config_path.exists():
        with open(sft_config_path) as f:
            adapter_config = json.load(f)
        adapter_config["dpo_beta"] = args.beta
        adapter_config["dpo_learning_rate"] = args.learning_rate
        adapter_config["dpo_iterations"] = args.iters
    else:
        adapter_config = {
            "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 2.0},
            "num_layers": args.lora_layers,
        }

    with open(output_path / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"Adapters saved to: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="DPO training with MLX")
    parser.add_argument("--model", required=True, help="Base model path or HF repo")
    parser.add_argument(
        "--adapter-path", default=None, help="Path to SFT LoRA adapters"
    )
    parser.add_argument("--data", required=True, help="Path to DPO pairs JSONL")
    parser.add_argument(
        "--output-adapter-path", required=True, help="Output path for DPO adapters"
    )
    parser.add_argument("--iters", type=int, default=50, help="Training iterations")
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6, help="Learning rate"
    )
    parser.add_argument(
        "--beta", type=float, default=0.1, help="DPO temperature parameter"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        dest="lora_layers",
        help="Number of LoRA layers",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
