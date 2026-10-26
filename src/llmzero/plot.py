"""Gráficos de linha em SVG puro, sem dependências.

Evita matplotlib só para desenhar curvas de perda e mantém a identidade visual
do portfólio (fundo neutro escuro, destaques em verde-limão e ciano).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from pathlib import Path

SURFACE, BORDER = "#151923", "#292F3A"
TEXT, MUTED = "#F5F7FA", "#A0A8B7"
# Cores das séries em ordem; o traço distingue séries quando as cores se repetem.
SERIES_COLORS = ["#A3FF12", "#00D9FF", "#F5F7FA", "#A0A8B7"]
DASHES = ["", "6 4", "2 3", "10 3 2 3"]


@dataclass(frozen=True)
class Series:
    label: str
    x: list[float]
    y: list[float]


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Marcas "redondas" que cobrem [lo, hi] por inteiro (a primeira <= lo, a última >= hi)."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first, last = math.floor(lo / step + 1e-9), math.ceil(hi / step - 1e-9)
    return [round(k * step, 10) for k in range(first, last + 1)]


def line_chart(
    series: list[Series], title: str, x_label: str, y_label: str, path: Path, y_max: float | None = None
) -> None:
    """Desenha as séries. ``y_max`` limita o eixo Y; pontos acima são cortados e a nota de rodapé avisa."""
    if not series or any(len(s.x) != len(s.y) or not s.x for s in series):
        raise ValueError("cada série precisa de x e y não vazios e do mesmo tamanho")
    legend_w = 190
    w, h = 760, 380
    left, right, top, bottom = 56, 24 + legend_w, 52, 56
    pw, ph = w - left - right, h - top - bottom
    xs = [v for s in series for v in s.x]
    ys = [v for s in series for v in s.y]
    x0, x1 = min(xs), max(xs)
    yt = _nice_ticks(min(ys), min(max(ys), y_max) if y_max is not None else max(ys))
    y0, y1 = yt[0], yt[-1]
    clipped = max(ys) > y1
    x1 = x1 if x1 > x0 else x0 + 1

    def px(v: float) -> float:
        return left + (v - x0) / (x1 - x0) * pw

    def py(v: float) -> float:
        return top + (1 - (v - y0) / (y1 - y0)) * ph

    font = 'font-family="Inter,Segoe UI,system-ui,sans-serif"'
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{escape(title)}">',
        f'<clipPath id="plot"><rect x="{left}" y="{top}" width="{pw}" height="{ph}"/></clipPath>',
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{SURFACE}" stroke="{BORDER}"/>',
        f'<text x="{left}" y="30" fill="{TEXT}" font-size="15" font-weight="600" {font}>{escape(title)}</text>',
    ]
    for t in yt:
        y = py(t)
        parts.append(f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{BORDER}"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" fill="{MUTED}" font-size="11" text-anchor="end" {font}>{t:g}</text>'
        )
    for t in _nice_ticks(x0, x1):
        if x0 <= t <= x1:
            parts.append(
                f'<text x="{px(t):.1f}" y="{top + ph + 18}" fill="{MUTED}" font-size="11" '
                f'text-anchor="middle" {font}>{t:g}</text>'
            )
    parts.append(
        f'<text x="{left + pw / 2}" y="{h - 16}" fill="{MUTED}" font-size="12" text-anchor="middle" {font}>'
        f"{escape(x_label)}</text>"
    )
    parts.append(
        f'<text transform="translate(18 {top + ph / 2}) rotate(-90)" fill="{MUTED}" font-size="12" '
        f'text-anchor="middle" {font}>{escape(y_label)}</text>'
    )
    if clipped:
        parts.append(
            f'<text x="{w - 16}" y="{h - 16}" fill="{MUTED}" font-size="11" text-anchor="end" {font}>'
            f"valores acima de {y1:g} cortados</text>"
        )
    legend_x = left + pw + 24
    for i, s in enumerate(series):
        color, dash = SERIES_COLORS[i % len(SERIES_COLORS)], DASHES[i % len(DASHES)]
        pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(s.x, s.y, strict=True))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{dash_attr} '
            'stroke-linejoin="round" stroke-linecap="round" clip-path="url(#plot)"/>'
        )
        ly = top + 6 + i * 20
        parts.append(
            f'<line x1="{legend_x}" x2="{legend_x + 22}" y1="{ly}" y2="{ly}" stroke="{color}" '
            f'stroke-width="2"{dash_attr}/>'
        )
        parts.append(
            f'<text x="{legend_x + 30}" y="{ly + 4}" fill="{TEXT}" font-size="11" {font}>{escape(s.label)}</text>'
        )
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")
