"""Small, local integration checks for the prepare/train/resume workflow."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from kenai import generate
from kenai.model import GPT, GPTConfig
from kenai.prepare import prepare
from kenai.tokenizer import CharTokenizer
from kenai.train import PRESETS, get_batch, train


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def arguments(self, **overrides) -> argparse.Namespace:
        values = dict(
            data=self.root / "corpus.txt", out=self.root / "run",
            preset=None, steps=2, batch_size=2, block_size=8,
            learning_rate=0.003, eval_every=2, eval_batches=1, seed=123,
            device="cpu", threads=1, resume=None,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def resume_arguments(self, checkpoint: Path, **overrides) -> argparse.Namespace:
        values = dict(
            data=None, out=checkpoint.parent, resume=checkpoint,
            block_size=None, batch_size=None, learning_rate=None, seed=None, steps=3,
        )
        values.update(overrides)
        return self.arguments(**values)

    def run_training(self, args: argparse.Namespace) -> None:
        tiny_preset = dict(block_size=8, n_embd=16, n_head=4, n_layer=1)
        with patch.dict(PRESETS, {"starter": tiny_preset}), redirect_stdout(io.StringIO()):
            train(args)

    def test_prepare_normalizes_deduplicates_and_protects_existing_output(self) -> None:
        writing = self.root / "writing"
        writing.mkdir()
        first = "I write slowly. Each sentence should earn its place.\n" * 35
        second = "Some days I change my mind. Then I try the idea again.\n" * 15
        (writing / "01.txt").write_bytes(("\ufeff  " + first.replace("\n", "\r\n") + "  ").encode("utf-8"))
        (writing / "02.md").write_text(first, encoding="utf-8")
        (writing / "03.md").write_text(second, encoding="utf-8")
        (writing / ".hidden.txt").write_text("Ignore this private note.", encoding="utf-8")
        (writing / "empty.txt").write_text("  \n", encoding="utf-8")
        (writing / "image.bin").write_bytes(b"\xff\xfe")
        output = writing / "combined.txt"

        metadata = prepare(writing, output)
        expected = first.strip() + "\n\n" + second.strip() + "\n"
        self.assertEqual(output.read_text(encoding="utf-8"), expected)
        self.assertEqual(metadata["documents"], 2)
        self.assertEqual(metadata["characters"], len(expected))
        self.assertEqual(metadata["sha256"], hashlib.sha256(expected.encode("utf-8")).hexdigest())
        self.assertEqual(json.loads(output.with_suffix(".txt.json").read_text()), metadata)
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(writing, output)
        self.assertEqual(output.read_text(encoding="utf-8"), expected)

    def test_checkpoint_reload_resume_and_corpus_integrity(self) -> None:
        source = self.root / "writing.txt"
        source.write_text(
            "I write slowly. Each sentence should earn its place.\n" * 40,
            encoding="utf-8",
        )
        corpus = self.root / "corpus.txt"
        prepare(source, corpus)
        self.run_training(self.arguments())
        checkpoint_path = self.root / "run" / "latest.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["step"], 2)

        def assert_cpu_tensors(value) -> int:
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device.type, "cpu")
                return 1
            if isinstance(value, dict):
                return sum(assert_cpu_tensors(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return sum(assert_cpu_tensors(item) for item in value)
            return 0

        self.assertGreater(assert_cpu_tensors(checkpoint), 10)
        tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
        restored = GPT(GPTConfig(**checkpoint["config"]))
        restored.load_state_dict(checkpoint["model"])
        self.assertEqual(asdict(restored.config), checkpoint["config"])
        prompt = "I write"
        with redirect_stdout(io.StringIO()) as output, patch(
            "sys.argv",
            ["kenai.generate", "--checkpoint", str(checkpoint_path), "--prompt", prompt,
             "--tokens", "4", "--top-k", "1", "--device", "cpu", "--threads", "1"],
        ):
            generate.main()
        generated = output.getvalue().removesuffix("\n")
        self.assertTrue(generated.startswith(prompt))
        self.assertEqual(len(generated), len(prompt) + 4)
        tokenizer.encode(generated)

        # Checkpointed RNG and optimizer state must reproduce uninterrupted CPU
        # training, including dropout and the random batch sequence.
        self.run_training(self.resume_arguments(checkpoint_path))
        resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        uninterrupted_path = self.root / "uninterrupted"
        self.run_training(self.arguments(out=uninterrupted_path, steps=3))
        uninterrupted = torch.load(uninterrupted_path / "latest.pt", map_location="cpu", weights_only=True)
        self.assertEqual(resumed["step"], 3)
        for name, parameter in resumed["model"].items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(parameter, uninterrupted["model"][name], rtol=0, atol=0)

        # A new output directory must receive a usable best checkpoint even
        # when no subsequent update beats the source run's previous best.
        branch = self.root / "resumed-elsewhere"
        self.run_training(self.resume_arguments(checkpoint_path, out=branch, steps=4))
        self.assertTrue((branch / "best.pt").is_file())
        self.assertEqual(torch.load(branch / "latest.pt", weights_only=True)["step"], 4)

        before = checkpoint_path.read_bytes()
        changed_corpus = self.root / "changed.txt"
        changed_corpus.write_text(corpus.read_text().replace("slowly", "quickly", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Corpus changed"):
            self.run_training(self.resume_arguments(checkpoint_path, data=changed_corpus, steps=4))
        self.assertEqual(checkpoint_path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "Run already exists"):
            self.run_training(self.arguments())
        self.assertEqual(checkpoint_path.read_bytes(), before)

    def test_batch_targets_shift_without_crossing_the_split(self) -> None:
        tokens = torch.arange(200, dtype=torch.long)
        boundary = 180
        generator = torch.Generator().manual_seed(7)
        for name, split in {"train": tokens[:boundary], "validation": tokens[boundary:]}.items():
            x, y = get_batch(split, block_size=8, batch_size=64, device="cpu", generator=generator)
            torch.testing.assert_close(y, x + 1)
            if name == "train":
                self.assertLess(y.max().item(), boundary)
            else:
                self.assertGreaterEqual(x.min().item(), boundary)
                self.assertLessEqual(y.max().item(), tokens[-1].item())


if __name__ == "__main__":
    unittest.main()
