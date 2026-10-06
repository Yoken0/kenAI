"""Train a small GPT from random weights, or continue one of your checkpoints."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .model import GPT, GPTConfig
from .runtime import choose_device, configure_threads
from .tokenizer import CharTokenizer


PRESETS = {
    "starter": dict(block_size=128, n_embd=128, n_head=4, n_layer=4),
    "studio": dict(block_size=256, n_embd=256, n_head=8, n_layer=6),
}


def get_batch(data, block_size, batch_size, device, generator):
    # y is x shifted one character ahead: every position predicts the next one.
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, splits, batch_size, device, batches, seed):
    model.eval()
    # Fixed held-out windows make loss at different steps comparable.
    generator = torch.Generator().manual_seed(seed)
    losses = {}
    for name, data in splits.items():
        total = 0.0
        for _ in range(batches):
            x, y = get_batch(data, model.config.block_size, batch_size, device, generator)
            _, loss = model(x, y)
            total += loss.item()
        losses[name] = total / batches
    model.train()
    return losses


def on_cpu(value):
    """Keep checkpoint tensors portable across CPU, Apple GPU, and CUDA."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: on_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [on_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(on_cpu(item) for item in value)
    return value


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(on_cpu(payload), temporary)
    temporary.replace(path)


def train(args):
    configure_threads(args.threads)
    device = choose_device(args.device)
    if args.steps < 1 or args.eval_every < 1 or args.eval_batches < 1:
        raise ValueError("steps, eval-every, and eval-batches must be positive")
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.preset or args.block_size is not None:
            raise ValueError("A resumed model keeps its architecture; omit --preset and --block-size.")
    data_path = args.data or Path(checkpoint["data_path"] if checkpoint else "data/shakespeare.txt")
    text = data_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if checkpoint and digest != checkpoint["data_sha256"]:
        raise ValueError("Corpus changed since this checkpoint. Start a new run for new writing.")
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"]) if checkpoint else CharTokenizer.from_text(text)
    if checkpoint:
        config = GPTConfig(**checkpoint["config"])
    else:
        settings = PRESETS[args.preset or "starter"].copy()
        if args.block_size is not None:
            settings["block_size"] = args.block_size
        config = GPTConfig(vocab_size=tokenizer.vocab_size, **settings)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    boundary = int(0.9 * len(tokens))
    splits = {"train": tokens[:boundary], "validation": tokens[boundary:]}
    if min(len(part) for part in splits.values()) <= config.block_size:
        raise ValueError(f"Need more text for a 90/10 split with context {config.block_size}. Add text or lower --block-size.")
    previous = checkpoint["training"] if checkpoint else {}
    seed = args.seed if args.seed is not None else previous.get("seed", 1337)
    batch_size = args.batch_size if args.batch_size is not None else previous.get("batch_size", 16)
    learning_rate = args.learning_rate if args.learning_rate is not None else previous.get("learning_rate", 3e-4)
    if batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("batch-size and learning-rate must be positive and finite")
    start_step = checkpoint["step"] if checkpoint else 0
    if args.steps <= start_step:
        raise ValueError(f"--steps is the total target; choose a value larger than the saved step {start_step}.")
    out = args.out or (args.resume.parent if args.resume else Path("runs/shakespeare"))
    if not checkpoint and any((out / name).exists() for name in ["latest.pt", "best.pt", "metrics.jsonl"]):
        raise ValueError(f"Run already exists at {out}; use --resume or choose a new --out folder.")
    if checkpoint and out.resolve() != args.resume.parent.resolve():
        if any((out / name).exists() for name in ["latest.pt", "best.pt", "metrics.jsonl"]):
            raise ValueError("The new output folder already contains a run. Choose an empty folder.")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        generator.set_state(checkpoint["batch_rng"])
        torch.set_rng_state(checkpoint["cpu_rng"])
    best = checkpoint["best_validation_loss"] if checkpoint else float("inf")
    # A new output folder starts its own best-checkpoint history. The source
    # run's best model may be different from the checkpoint being resumed.
    if checkpoint and not (out / "best.pt").exists():
        best = float("inf")
    print(f"Device: {device} | Parameters: {model.num_parameters():,} | Characters: {len(tokens):,}", flush=True)
    print(f"Training: {boundary:,} characters | Validation: {len(tokens) - boundary:,} | Context: {config.block_size}", flush=True)
    print(f"Run: {out} | Step {start_step} → {args.steps}. Ctrl+C saves your progress.", flush=True)
    training = {"seed": seed, "batch_size": batch_size, "learning_rate": learning_rate}
    step = start_step
    started = time.monotonic()

    def payload():
        return {
            "format_version": 1, "config": asdict(config), "tokenizer": tokenizer.to_dict(),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
            "best_validation_loss": best, "data_path": str(data_path), "data_sha256": digest,
            "training": training, "batch_rng": generator.get_state(), "cpu_rng": torch.get_rng_state(),
        }

    def record_evaluation():
        nonlocal best
        losses = evaluate(model, splits, batch_size, device, args.eval_batches, seed + 1)
        if not all(math.isfinite(loss) for loss in losses.values()):
            raise ValueError("Loss is not finite. Try a lower learning rate in a new run.")
        improved = losses["validation"] < best
        if improved:
            best = losses["validation"]
        row = {"step": step, "train_loss": losses["train"], "validation_loss": losses["validation"],
               "elapsed_seconds": round(time.monotonic() - started, 2)}
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        saved = payload()
        save_checkpoint(out / "latest.pt", saved)
        if improved:
            save_checkpoint(out / "best.pt", saved)
        print(f"step {step:5d} | train {losses['train']:.4f} | validation {losses['validation']:.4f} | {row['elapsed_seconds']:.1f}s", flush=True)

    try:
        if not checkpoint or not (out / "best.pt").exists():
            record_evaluation()
        model.train()
        while step < args.steps:
            x, y = get_batch(splits["train"], config.block_size, batch_size, device, generator)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if not torch.isfinite(loss).item():
                raise ValueError("Training loss is not finite. Try a lower learning rate in a new run.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                record_evaluation()
    except KeyboardInterrupt:
        save_checkpoint(out / "latest.pt", payload())
        print(f"\nStopped at step {step}; saved {out / 'latest.pt'}", flush=True)
        return
    print(f"Saved {out / 'latest.pt'} (resume) and best.pt (generate).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--steps", type=int, default=500, help="Total optimizer steps, including already saved steps")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Training error: {exc}\n")


if __name__ == "__main__":
    main()
