"""A character vocabulary learned only from your training text."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


class CharTokenizer:
    """Map each Unicode character to an integer and back again.

    Character tokenization is intentionally simple: no external tokenizer or
    pretrained vocabulary is needed, and every generated character can be
    traced back to the training text.
    """

    def __init__(self, chars: Sequence[str]) -> None:
        if not chars:
            raise ValueError("The vocabulary must contain at least one character.")
        if any(not isinstance(char, str) or len(char) != 1 for char in chars):
            raise ValueError("Every vocabulary entry must be a single character.")
        self.chars = sorted(set(chars))
        self._char_to_id = {char: index for index, char in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(text))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - self._char_to_id.keys())
        if unknown:
            display = ", ".join(repr(char) for char in unknown[:10])
            if len(unknown) > 10:
                display += ", ..."
            raise ValueError(
                f"Text contains characters outside the training vocabulary: {display}. "
                "Use characters present in your training text."
            )
        return [self._char_to_id[char] for char in text]

    def decode(self, ids: Iterable[int]) -> str:
        characters = []
        for token_id in ids:
            if (
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or not 0 <= token_id < self.vocab_size
            ):
                raise ValueError(f"Invalid character token ID: {token_id!r}.")
            characters.append(self.chars[token_id])
        return "".join(characters)

    def to_dict(self) -> dict[str, list[str]]:
        return {"chars": list(self.chars)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CharTokenizer":
        if not isinstance(data, Mapping) or not isinstance(data.get("chars"), list):
            raise ValueError("A saved tokenizer must contain a 'chars' list.")
        chars = data["chars"]
        tokenizer = cls(chars)
        # IDs are stored in datasets and checkpoints. Reordering a saved
        # vocabulary here would silently change what those IDs mean.
        if tokenizer.chars != chars:
            raise ValueError("A saved tokenizer's characters must be sorted and unique.")
        return tokenizer
