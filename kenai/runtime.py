"""Device selection shared by training and generation."""

import torch


def choose_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device auto or --device cpu.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple GPU is unavailable to this process. Use --device cpu.")
    return requested


def configure_threads(threads: int):
    if threads < 1:
        raise ValueError("--threads must be at least 1")
    torch.set_num_threads(threads)
