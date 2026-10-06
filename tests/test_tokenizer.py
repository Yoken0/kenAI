import json
import unittest

from kenai.tokenizer import CharTokenizer


class CharTokenizerTests(unittest.TestCase):
    def test_unicode_round_trip_and_stable_ids(self) -> None:
        text = "Hey, café!\nI’m Ken. 🐈"
        tokenizer = CharTokenizer.from_text(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.chars, sorted(set(text)))
        self.assertEqual(tokenizer.vocab_size, len(set(text)))
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")

    def test_checkpoint_round_trip_preserves_token_ids(self) -> None:
        tokenizer = CharTokenizer.from_text("cab cab\n")
        saved = json.loads(json.dumps(tokenizer.to_dict()))
        loaded = CharTokenizer.from_dict(saved)
        self.assertEqual(loaded.encode("cab\n"), tokenizer.encode("cab\n"))
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            CharTokenizer.from_dict({"chars": ["b", "a"]})

    def test_unknown_characters_are_reported(self) -> None:
        tokenizer = CharTokenizer.from_text("abc")
        with self.assertRaisesRegex(ValueError, "outside the training vocabulary: 'z'"):
            tokenizer.encode("abz")

    def test_invalid_vocabulary_and_ids(self) -> None:
        for chars in ([], [""], ["word"], [1]):
            with self.subTest(chars=chars), self.assertRaises(ValueError):
                CharTokenizer(chars)
        tokenizer = CharTokenizer.from_text("ab")
        for token_id in (-1, 2, 1.5, True):
            with self.subTest(token_id=token_id), self.assertRaises(ValueError):
                tokenizer.decode([token_id])


if __name__ == "__main__":
    unittest.main()
