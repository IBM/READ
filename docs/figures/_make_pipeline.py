# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Render the README pipeline figure.

Run from repo root:
    python docs/figures/_make_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "pipeline.png"

INK = "#111111"
MUTED = "#6a6a6a"
HAIRLINE = "#bdbdbd"
ACCENT = "#2f6f3b"


def box(ax, x, y, w, h, *, edge=INK, lw=1.0, fill="white"):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw,
                           facecolor=fill, edgecolor=edge))


def label(ax, x, y, text, *, ts=10, weight="normal", color=INK,
          ha="left", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=ts, color=color, weight=weight)


def arrow(ax, x1, y1, x2, y2, *, color=INK, lw=1.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
    ))


def render():
    W, H = 9.8, 2.9
    fig, ax = plt.subplots(figsize=(W, H), dpi=200)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # Title
    label(ax, W / 2, H - 0.16,
          "READ: per-instance causal erasure",
          ts=11, weight="bold", color=INK, ha="center")

    # Inputs ----------------------------------------------------------------
    in_w, in_h = 1.35, 0.62
    in_x = 0.12
    p_y = 1.50
    s_y = 0.42

    box(ax, in_x, p_y, in_w, in_h)
    label(ax, in_x + 0.14, p_y + in_h * 0.62, "Policy", ts=10.2, weight="bold")
    label(ax, in_x + 0.14, p_y + in_h * 0.28, "$r_1,\\dots,r_k$", ts=9.0, color=MUTED)

    box(ax, in_x, s_y, in_w, in_h)
    label(ax, in_x + 0.14, s_y + in_h * 0.62, "Scenario", ts=10.2, weight="bold")
    label(ax, in_x + 0.14, s_y + in_h * 0.28, "violates $r^*$", ts=9.0, color=MUTED)

    # Two input arrows meet at a point. From there a single horizontal
    # arrow continues into the perturbation column (drawn further below).
    cx_merge = in_x + in_w + 0.40
    cy_merge = (p_y + s_y + in_h) / 2

    # arrows from inputs end just before the merge point (no arrowhead clutter)
    ax.add_patch(FancyArrowPatch(
        (in_x + in_w, p_y + in_h / 2), (cx_merge, cy_merge),
        arrowstyle="-", linewidth=1.1, color=INK, shrinkA=0, shrinkB=0,
    ))
    ax.add_patch(FancyArrowPatch(
        (in_x + in_w, s_y + in_h / 2), (cx_merge, cy_merge),
        arrowstyle="-", linewidth=1.1, color=INK, shrinkA=0, shrinkB=0,
    ))
    # filled junction dot
    ax.plot([cx_merge], [cy_merge], "o", markersize=4.5, color=INK)

    # Perturbation rows -----------------------------------------------------
    pt_x = cx_merge + 0.55
    verdict_col_x = pt_x
    label_col_x = pt_x + 0.50

    rows = [
        ("Full policy",                 "V", INK,    "baseline"),
        ("Empty policy",                "?", MUTED,  "prior probe"),
        ("Drop $r^*$",                  "C", ACCENT, "should flip"),
        ("Drop $r_i$, $i{\\ne}*$",      "V", ACCENT, "should stay"),
    ]
    row_h = 0.42
    row_gap = 0.06
    rows_total = len(rows) * row_h + (len(rows) - 1) * row_gap
    rows_top = cy_merge + rows_total / 2

    row_centers = []
    for i, (lbl, verdict, vcol, note) in enumerate(rows):
        ry = rows_top - (i + 1) * row_h - i * row_gap
        cy = ry + row_h / 2
        row_centers.append(cy)

        label(ax, verdict_col_x, cy, verdict, ts=13.5, weight="bold",
              color=vcol, ha="left", va="center")
        label(ax, label_col_x, cy + 0.09, lbl, ts=10, weight="bold")
        label(ax, label_col_x, cy - 0.10, note, ts=8.2, color=MUTED)

    # Arrow from the merge dot to the perturbation column
    arrow(ax, cx_merge + 0.05, cy_merge, pt_x - 0.08, cy_merge, lw=1.1)

    # Right edge of perturbation text (arrow fan-in origin)
    pt_right = pt_x + 1.55

    # Judge box
    judge_w = 1.30
    judge_h = 1.10
    judge_x = pt_right + 1.20
    judge_y = cy_merge - judge_h / 2

    # FAN-IN: one arrow per perturbation row -> left edge of judge
    # ink color = data flow; MUTED on outputs only
    fan_target_x = judge_x - 0.02
    fan_target_y = cy_merge
    for cy in row_centers:
        arrow(ax, pt_right + 0.04, cy, fan_target_x, fan_target_y,
              color=INK, lw=0.9)
    # label sits on the centre fan-arrow with a white bbox to clear the line
    label_x = (pt_right + judge_x) / 2 + 0.05
    ax.text(label_x, cy_merge + 0.03, "$k{+}2$ judge calls",
            ha="center", va="center", fontsize=8.4, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none"))

    box(ax, judge_x, judge_y, judge_w, judge_h, lw=1.2)
    label(ax, judge_x + judge_w / 2, judge_y + judge_h * 0.60,
          "LLM judge", ts=10.5, weight="bold", ha="center")
    label(ax, judge_x + judge_w / 2, judge_y + judge_h * 0.30,
          "any provider", ts=8.5, color=MUTED, ha="center")

    # Outputs ---------------------------------------------------------------
    out_x = judge_x + judge_w + 0.55
    out_w = 1.65
    out_h = 0.50
    out_gap = 0.13
    out_centers_y = [cy_merge + (out_h + out_gap),
                     cy_merge,
                     cy_merge - (out_h + out_gap)]
    out_labels = ["5 metrics", "4 failure modes", "per-instance flags"]

    for cy, txt in zip(out_centers_y, out_labels):
        y = cy - out_h / 2
        box(ax, out_x, y, out_w, out_h)
        label(ax, out_x + out_w / 2, cy, txt, ts=9.3,
              weight="bold", ha="center")
        arrow(ax, judge_x + judge_w + 0.02, cy_merge,
              out_x - 0.02, cy, lw=0.9, color=MUTED)

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    render()
