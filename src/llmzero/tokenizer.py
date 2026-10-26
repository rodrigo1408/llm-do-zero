"""Tokenização em nível de caractere.

Cada caractere distinto do corpus vira um id inteiro. O id 0 é reservado ao
token de fronteira ``.``, que marca início e fim de cada sequência — é assim
que o modelo aprende quando um nome começa e quando termina.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

BOUNDARY = "."


class TokenizerError(ValueError):
    """Texto incompatível com o vocabulário."""


class CharTokenizer:
    def __init__(self, chars: Iterable[str]):
        chars = list(chars)
        if not chars or chars[0] != BOUNDARY:
            raise TokenizerError(f"o vocabulário deve começar pelo token de fronteira {BOUNDARY!r}")
        if len(set(chars)) != len(chars):
            raise TokenizerError("vocabulário com caracteres repetidos")
        self.itos: list[str] = chars
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}

    @classmethod
    def fit(cls, texts: Iterable[str]) -> CharTokenizer:
        """Constrói o vocabulário a partir do corpus (ordem determinística)."""
        seen: set[str] = set()
        for text in texts:
            if BOUNDARY in text:
                raise TokenizerError(f"o corpus não pode conter o token de fronteira {BOUNDARY!r}: {text!r}")
            seen.update(text)
        return cls([BOUNDARY, *sorted(seen)])

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[c] for c in text]
        except KeyError as exc:
            raise TokenizerError(f"caractere fora do vocabulário: {exc.args[0]!r}") from None

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_json(self) -> str:
        return json.dumps({"chars": self.itos}, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> CharTokenizer:
        return cls(json.loads(payload)["chars"])

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CharTokenizer:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
