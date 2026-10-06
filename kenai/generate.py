"""Continue a prompt with weights that you trained yourself."""

import argparse
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .runtime import choose_device, configure_threads
from .tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/shakespeare/best.pt"))
    parser.add_argument("--prompt", default="", help="Text to continue; use characters found in your corpus")
    parser.add_argument("--tokens", type=int, default=400, help="New characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        configure_threads(args.threads)
        device = choose_device(args.device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
        model = GPT(GPTConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        torch.manual_seed(args.seed)
        prompt = args.prompt or ("\n" if "\n" in tokenizer.chars else tokenizer.chars[0])
        tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        result = model.generate(tokens, args.tokens, args.temperature, args.top_k)
        print(tokenizer.decode(result[0].tolist()))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Generation error: {exc}\n")


if __name__ == "__main__":
    main()
