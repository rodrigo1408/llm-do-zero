"""Baixa a lista oficial de municípios brasileiros (IBGE) e grava data/municipios.txt.

Fonte: API de Localidades do IBGE (dados públicos)
https://servicodados.ibge.gov.br/api/docs/localidades

O arquivo gerado já está versionado; este script existe para que a origem dos
dados seja reproduzível e auditável. Usa apenas a biblioteca padrão.
"""

from __future__ import annotations

import gzip
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path

URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios?view=nivelado"
OUT = Path(__file__).resolve().parent.parent / "data" / "municipios.txt"


def normalize(name: str) -> str:
    """Forma canônica usada no corpus: NFC, minúsculas, espaços simples."""
    return " ".join(unicodedata.normalize("NFC", name).lower().split())


def main() -> int:
    with urllib.request.urlopen(URL, timeout=30) as resp:  # noqa: S310 - URL fixa (https)
        raw = resp.read()
        # A API responde comprimida mesmo sem Accept-Encoding.
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    names = sorted({normalize(item["municipio-nome"]) for item in payload})
    if len(names) < 5000:
        print(f"resposta inesperada do IBGE: {len(names)} nomes", file=sys.stderr)
        return 1
    OUT.write_text("\n".join(names) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(payload)} municípios, {len(names)} nomes distintos -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
