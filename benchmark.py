import time
import argparse

import torch
from transformers import GPT2TokenizerFast

from generate import load_model as load_gen_model, generate
from speculative_generate import load_model as load_spec_model, speculative_generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-path", type=str, default="checkpoints/target.pth")
    parser.add_argument("--draft-model-path", type=str, default="checkpoints/draft.pth")
    parser.add_argument("--draft-length", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer)

    # ===== load 一次，不计入 =====
    print("Loading models...")
    target_model,_ = load_gen_model(args.target_model_path, device)
    draft_model = load_spec_model(args.draft_model_path, device)

    print("=" * 60)
    print("Speculative Decoding Benchmark")
    print("=" * 60)
    print(f"prompt:           {args.prompt}")
    print(f"max_new_tokens:   {args.max_new_tokens}")
    print(f"draft_length:     {args.draft_length}")
    print(f"device:           {device}")
    print(f"warmup / runs:    {args.warmup} / {args.runs}")
    print("-" * 60)

    # ===== 预热 =====
    print("Warming up...")
    for _ in range(args.warmup):
        generate(target_model, tokenizer, args.prompt, device,
                 max_new_tokens=args.max_new_tokens)
    for _ in range(args.warmup):
        speculative_generate(
            target_model, draft_model,
            args.target_model_path, args.draft_model_path,
            args.draft_length, args.prompt, device,
            args.max_new_tokens, args.tokenizer,
        )

    # ===== 普通生成 =====
    print("Running plain generation...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.runs):
        generate(target_model, tokenizer, args.prompt, device,
                 max_new_tokens=args.max_new_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    t_plain = (t1 - t0) / args.runs
    print(f"plain:            {t_plain:.4f} s")

    # ===== 投机生成 =====
    print("Running speculative generation...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.runs):
        speculative_generate(
            target_model, draft_model,
            args.target_model_path, args.draft_model_path,
            args.draft_length, args.prompt, device,
            args.max_new_tokens, args.tokenizer,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    t_spec = (t1 - t0) / args.runs
    print(f"speculative:      {t_spec:.4f} s")

    # ===== 加速比 =====
    speedup = t_plain / t_spec
    print("-" * 60)
    print(f"speedup:          {speedup:.3f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
