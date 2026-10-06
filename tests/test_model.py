import unittest

import torch

from kenai.model import GPT, GPTConfig


class GPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        torch.manual_seed(42)

    def tiny_model(self, **overrides: int | float) -> GPT:
        config = dict(vocab_size=7, block_size=8, n_embd=16, n_head=4, n_layer=2, dropout=0.0)
        config.update(overrides)
        return GPT(GPTConfig(**config))

    def test_future_tokens_cannot_change_earlier_predictions(self) -> None:
        model = self.tiny_model().eval()
        original = torch.tensor([[0, 1, 2, 3, 4, 5]])
        changed = torch.tensor([[0, 1, 2, 6, 0, 1]])
        with torch.no_grad():
            original_logits, _ = model(original)
            changed_logits, _ = model(changed)
        torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3])
        self.assertFalse(torch.allclose(original_logits[:, 3:], changed_logits[:, 3:]))

    def test_next_token_loss_has_finite_gradients(self) -> None:
        model = self.tiny_model()
        inputs = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        targets = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 6]])
        logits, loss = model(inputs, targets)
        self.assertEqual(logits.shape, (2, 4, 7))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.token_embedding.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.num_parameters(), sum(p.numel() for p in model.parameters()))

    def test_optimizer_learns_a_repeating_sequence(self) -> None:
        model = self.tiny_model(n_layer=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        inputs = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])
        targets = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]])
        initial_loss = model(inputs, targets)[1].item()
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
        final_loss = model(inputs, targets)[1].item()
        self.assertLess(final_loss, initial_loss * 0.25)

    def test_generation_crops_context_and_preserves_prompt_and_mode(self) -> None:
        model = self.tiny_model(block_size=4, dropout=0.2)
        prompt = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]])
        output = model.generate(prompt, max_new_tokens=5, top_k=40)
        self.assertEqual(output.shape, (2, 11))
        torch.testing.assert_close(output[:, :6], prompt)
        self.assertTrue(((output >= 0) & (output < model.config.vocab_size)).all())
        self.assertTrue(model.training)
        model.eval()
        model.generate(prompt, max_new_tokens=1, top_k=None)
        self.assertFalse(model.training)

    def test_top_one_generation_is_deterministic(self) -> None:
        model = self.tiny_model(dropout=0.2)
        prompt = torch.tensor([[0, 1]])
        first = model.generate(prompt, max_new_tokens=8, top_k=1)
        second = model.generate(prompt, max_new_tokens=8, top_k=1)
        torch.testing.assert_close(first, second)

    def test_invalid_configuration_and_generation_options(self) -> None:
        for invalid in (
            {"vocab_size": 0}, {"block_size": 0}, {"n_embd": 15},
            {"n_head": 0}, {"n_layer": -1}, {"dropout": 1.0}, {"dropout": float("nan")},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.tiny_model(**invalid)
        model = self.tiny_model()
        prompt = torch.tensor([[0]])
        for invalid in (
            {"max_new_tokens": -1}, {"max_new_tokens": 1.5},
            {"temperature": 0}, {"temperature": float("nan")}, {"top_k": 0},
        ):
            options = dict(max_new_tokens=1)
            options.update(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                model.generate(prompt, **options)
        with self.assertRaisesRegex(ValueError, "exceeds block_size"):
            model(torch.zeros((1, 9), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
