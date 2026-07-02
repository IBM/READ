# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Render the failure-mode decision-tree figure.

Mirrors src/failure_modes.py (violation branch): four verdict-equality
predicates route every (y_P, y_0, y_{P-r*}, y_{P-i}) tuple into one of
Prior-Based, Policy-Degraded, Confused, or Faithful. The happy path runs
top-to-bottom through the centre column; failure outcomes branch right.

Run from repo root:
    python docs/figures/_make_failure_modes.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parent / "failure_modes.png"

INK = "#111111"
MUTED = "#6a6a6a"
HAIRLINE = "#bdbdbd"
GOOD = "#356b3a"
WARN = "#a8741c"
ALERT = "#8e3b30"


def rect(ax, x, y, w, h, *, edge=INK, lw=1.0, fill="white"):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw,
                           facecolor=fill, edgecolor=edge))


def diamond(ax, cx, cy, w, h, *, edge=INK, lw=1.0, fill="white"):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy),
           (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=lw,
                         facecolor=fill, edgecolor=edge))


def text(ax, x, y, s, *, ts=10, weight="normal", color=INK,
         ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=ts, color=color, weight=weight)


def arrow(ax, x1, y1, x2, y2, *, color=INK, lw=1.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
    ))


def render():
    W, H = 11.2, 5.7
    fig, ax = plt.subplots(figsize=(W, H), dpi=200)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # Title (no subtitle, no footer)
    text(ax, W / 2, 5.40,
         "Failure-mode decision tree  (violation instances)",
         ts=12, weight="bold")

    # Decision-chain centre column, all diamonds same width
    cx = 5.6
    dw, dh = 2.85, 0.80

    # y-positions: D1, D2, D3, D4, Faithful
    y_d1, y_d2, y_d3, y_d4 = 4.55, 3.65, 2.75, 1.85
    y_faithful = 0.55

    # Input tuple (top-left, vertically centred on first diamond)
    inp_w, inp_h = 2.85, 1.15
    inp_x = 0.20
    inp_y = y_d1 - inp_h / 2
    rect(ax, inp_x, inp_y, inp_w, inp_h, lw=1.0, fill="#f4f4f4")
    text(ax, inp_x + inp_w / 2, inp_y + inp_h - 0.22,
         "Input verdict tuple", ts=10, weight="bold")
    text(ax, inp_x + inp_w / 2, inp_y + inp_h / 2 - 0.10,
         "$y_P$: full     $y_0$: empty\n"
         "$y_{P\\setminus r^*}$: drop violated clause $r^*$\n"
         "$y_{P\\setminus i}$: drop each other clause",
         ts=8.6, color=INK)

    # Input -> D1
    arrow(ax, inp_x + inp_w + 0.04, y_d1, cx - dw / 2 - 0.04, y_d1, lw=1.1)

    # Decisions (centre column)
    decisions = [
        (y_d1, "$y_P\\;=\\;y_0$ ?"),
        (y_d2, "$y_P{=}C$   and   $y_0{=}V$ ?"),
        (y_d3, "$y_P{=}V$   and   $y_{P\\setminus r^*}{=}C$ ?"),
        (y_d4, "all $y_{P\\setminus i}{=}V\\;\\;(i{\\ne}*)$ ?"),
    ]
    for cy, lbl in decisions:
        diamond(ax, cx, cy, dw, dh, lw=1.0, fill="white")
        text(ax, cx, cy, lbl, ts=9.7, weight="bold")

    # Outcomes (right column, vertically aligned with decisions)
    out_x = 8.85
    out_w, out_h = 2.45, 0.78
    outcomes = [
        (y_d1, "Prior-Based",            "verdict identical w/ and w/o policy",   ALERT),
        (y_d2, "Policy-Degraded",        "right w/o policy, wrong with it",       ALERT),
        (y_d3, "Confused (mis-anchored)", "$r^*$ removal did not flip the verdict", WARN),
        (y_d4, "Confused (collateral)",   "non-$r^*$ removal also flipped it",      WARN),
    ]
    for cy, title, subtitle, color in outcomes:
        y = cy - out_h / 2
        rect(ax, out_x, y, out_w, out_h, edge=color, lw=1.4)
        text(ax, out_x + out_w / 2, cy + 0.16, title,
             ts=10.5, weight="bold", color=color)
        text(ax, out_x + out_w / 2, cy - 0.16, subtitle,
             ts=8.5, color=MUTED)

    # Faithful (bottom of centre column), light green tint to mark as terminus
    fy = y_faithful - out_h / 2
    rect(ax, cx - dw / 2, fy, dw, out_h, edge=GOOD, lw=1.6, fill="#eaf3eb")
    text(ax, cx, y_faithful + 0.16, "Faithful",
         ts=11, weight="bold", color=GOOD)
    text(ax, cx, y_faithful - 0.16, "correct and clause-grounded",
         ts=8.8, color=MUTED)

    # Vertical arrows between decisions, labels coloured MUTED for consistency
    def vertical(cy_from, cy_to, edge_label):
        x_from_bottom = cy_from - dh / 2
        x_to_top = cy_to + dh / 2
        arrow(ax, cx, x_from_bottom - 0.02, cx, x_to_top + 0.02, lw=1.1)
        ax.text(cx + 0.22, (x_from_bottom + x_to_top) / 2,
                edge_label, ha="left", va="center", fontsize=9.5,
                weight="bold", color=MUTED,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none"))

    vertical(y_d1, y_d2, "no")
    vertical(y_d2, y_d3, "no")
    vertical(y_d3, y_d4, "yes")
    # final vertical: D4 -> Faithful (yes), coloured GOOD to match outcome
    arrow(ax, cx, y_d4 - dh / 2 - 0.02, cx, y_faithful + out_h / 2 + 0.02,
          lw=1.3, color=GOOD)
    ax.text(cx + 0.22, (y_d4 - dh / 2 + y_faithful + out_h / 2) / 2,
            "yes", ha="left", va="center", fontsize=9.5,
            weight="bold", color=GOOD,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none"))

    # Horizontal arrows: decision -> outcome, harmonised typography
    def horizontal(cy, branch_label, color):
        arrow(ax, cx + dw / 2 + 0.02, cy, out_x - 0.02, cy,
              lw=1.1, color=color)
        ax.text((cx + dw / 2 + out_x) / 2, cy + 0.17,
                branch_label, ha="center", va="center", fontsize=9.5,
                weight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                          edgecolor="none"))

    horizontal(y_d1, "yes", ALERT)
    horizontal(y_d2, "yes", ALERT)
    horizontal(y_d3, "no",  WARN)
    horizontal(y_d4, "no",  WARN)

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    render()
