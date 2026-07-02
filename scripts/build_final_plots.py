#!/usr/bin/env python3
# Copyright contributors to the READ project
# SPDX-License-Identifier: Apache-2.0
"""Generate publication-quality scatter plots for READ NeurIPS figures."""

from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "draft" / "figures" / "generated"

COLORS = {
    "gpt-oss-safeguard-20b": "#D4A106",
    "granite-guardian-3.3-8b": "#C44E52",
    "llama-3-1-8b-instruct": "#4C72B0",
    "llama-3-3-70b-instruct": "#55A868",
    "llama-guard-3-8b": "#8172B2",
}

MARKERS = {
    "gpt-oss-safeguard-20b": "o",
    "granite-guardian-3.3-8b": "s",
    "llama-3-1-8b-instruct": "^",
    "llama-3-3-70b-instruct": "D",
    "llama-guard-3-8b": "p",
}


def _base_style():
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "medium",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.edgecolor": "#374151",
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    })


def plot_fig3b():
    df = pd.read_csv(GEN / "fig3_taxonomy_scatter.csv")
    _base_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    ax.set_xlim(85.5, 100.5)
    ax.set_ylim(-1, 28)

    ax.fill_between([95, 100.5], 15, 28, color="#dcfce7", alpha=0.65, zorder=0)
    ax.fill_between([85.5, 95], 15, 28, color="#fee2e2", alpha=0.5, zorder=0)
    ax.fill_between([85.5, 95], -1, 15, color="#fff7ed", alpha=0.55, zorder=0)
    ax.fill_between([95, 100.5], -1, 15, color="#f1f5f9", alpha=0.6, zorder=0)

    ax.axvline(95, color="#9ca3af", ls="--", lw=1.0, zorder=1)
    ax.axhline(15, color="#9ca3af", ls="--", lw=1.0, zorder=1)

    ax.text(97.8, 26.2, "Faithful", fontsize=11, fontstyle="italic",
            color="#16a34a", ha="center", fontweight="bold", alpha=0.85)
    ax.text(90.2, 26.2, "Confused", fontsize=11, fontstyle="italic",
            color="#dc2626", ha="center", fontweight="bold", alpha=0.85)
    ax.text(90.0, 1.2, "Policy-sensitive", fontsize=10.5, fontstyle="italic",
            color="#ea580c", ha="center", fontweight="bold", alpha=0.8)
    ax.text(97.8, 1.2, "Independent", fontsize=10.5, fontstyle="italic",
            color="#64748b", ha="center", fontweight="bold", alpha=0.8)

    label_offsets = {
        "gpt-oss-safeguard-20b": (0.45, 0.8),
        "granite-guardian-3.3-8b": (0.45, 0.7),
        "llama-3-1-8b-instruct": (0.45, -1.3),
        "llama-3-3-70b-instruct": (0.45, 0.75),
        "llama-guard-3-8b": (0.45, -1.1),
    }

    for _, r in df.iterrows():
        m = r["model_key"]
        ax.scatter(r["asr_pct"], r["csr_pct"],
                   s=140 if MARKERS[m] != "p" else 160,
                   c=COLORS[m], marker=MARKERS[m],
                   edgecolor="white", linewidth=1.4, zorder=5)
        dx, dy = label_offsets[m]
        ax.annotate(r["model_label"], (r["asr_pct"], r["csr_pct"]),
                    xytext=(dx, dy), textcoords="offset fontsize",
                    fontsize=9.5, fontweight="bold", color="#1e293b",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88),
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    ax.set_xlabel("Ablation Specificity Rate (ASR, %)", labelpad=8)
    ax.set_ylabel("Clause Sensitivity Rate (CSR, %)", labelpad=8)
    ax.set_title("IBM HR: CSR-ASR diagnostic space", pad=12)

    ax.set_xtick = ax.set_xticks([86, 88, 90, 92, 94, 96, 98, 100])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.grid(True, color="#e5e7eb", lw=0.6, alpha=0.7, zorder=0)

    fig.tight_layout()
    fig.savefig(GEN / "fig3_taxonomy_scatter.pdf", bbox_inches="tight")
    fig.savefig(GEN / "fig3_taxonomy_scatter.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  fig3b done")


def plot_fig4():
    df = pd.read_csv(GEN / "fig4_acc_vs_pir.csv")
    _base_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    ax.set_xlim(7, 37)
    ax.set_ylim(61, 93)

    ax.fill_betweenx([61, 93], 7, 15, color="#ecfdf5", alpha=0.5, zorder=0)

    x = df["pir_pct"].to_numpy()
    y = df["accuracy_pct"].to_numpy()
    coeffs = np.polyfit(x, y, 1)
    xs = np.linspace(7, 37, 100)
    ys = coeffs[0] * xs + coeffs[1]
    ax.plot(xs, ys, color="#9ca3af", lw=1.6, ls=(0, (6, 3)), zorder=2, alpha=0.7)

    label_offsets = {
        "gpt-oss-safeguard-20b": (0.55, 1.0),
        "granite-guardian-3.3-8b": (0.55, -1.4),
        "llama-3-1-8b-instruct": (-6.0, 1.1),
        "llama-3-3-70b-instruct": (0.55, 0.8),
        "llama-guard-3-8b": (0.55, -1.1),
    }

    for _, r in df.iterrows():
        m = r["model_key"]
        ax.scatter(r["pir_pct"], r["accuracy_pct"],
                   s=150 if MARKERS[m] != "p" else 170,
                   c=COLORS[m], marker=MARKERS[m],
                   edgecolor="white", linewidth=1.4, zorder=5)
        dx, dy = label_offsets[m]
        ax.annotate(r["model_label"], (r["pir_pct"], r["accuracy_pct"]),
                    xytext=(dx, dy), textcoords="offset fontsize",
                    fontsize=9.5, fontweight="bold", color="#1e293b",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88),
                    arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.7)
                    if abs(dx) > 3 else None,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    ax.annotate("High accuracy,\nlow policy dependence",
                xy=(11, 90.5), fontsize=9.5, color="#166534",
                fontstyle="italic", fontweight="medium",
                bbox=dict(boxstyle="round,pad=0.4", fc="#ecfdf5", ec="#bbf7d0", alpha=0.95))

    ax.annotate("Policy-reactive\nbut less accurate",
                xy=(29, 84), fontsize=9.5, color="#64748b",
                fontstyle="italic", fontweight="medium")

    ax.set_xlabel("Policy Influence Rate (PIR, %)", labelpad=8)
    ax.set_ylabel("Accuracy (%)", labelpad=8)
    ax.set_title("IBM HR: accuracy vs policy dependence", pad=12)

    ax.set_xticks([10, 15, 20, 25, 30, 35])
    ax.set_yticks([65, 70, 75, 80, 85, 90])
    ax.grid(True, color="#e5e7eb", lw=0.6, alpha=0.7, zorder=0)

    fig.tight_layout()
    fig.savefig(GEN / "fig4_acc_vs_pir.pdf", bbox_inches="tight")
    fig.savefig(GEN / "fig4_acc_vs_pir.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  fig4 done")


if __name__ == "__main__":
    print("Building final plots...")
    plot_fig3b()
    plot_fig4()
    print("Done.")
