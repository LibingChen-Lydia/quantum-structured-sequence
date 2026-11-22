"""Aggregate Module A runs into docs/results_module_a.md and docs/figures/*.png."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "module_a"
DOCS = ROOT / "docs"
FIG = DOCS / "figures"

# validated categorical palette (fixed order), surfaces and ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
})


def load(exp: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((OUT / exp).glob("*_s*.json"))]


def agg(runs: list[dict], key=lambda r: r["test_mse"]) -> dict:
    groups = defaultdict(list)
    for r in runs:
        groups[(r["tag"], r["n_train"])].append(r)
    out = {}
    for k, rs in groups.items():
        v = np.array([key(r) for r in rs])
        out[k] = {"mean": v.mean(), "std": v.std(), "n": len(v), "runs": rs}
    return out


def fmt(a: dict) -> str:
    return f"{a['mean']:.4f} ± {a['std']:.4f}"


LABEL = {
    "raw_mlp": "MLP on raw (A, φ)", "fourier_mlp": "MLP on Fourier (A, cos φ, sin φ)",
    "explicit": "Explicit interaction (ridge)", "quantum_fixed": "Fixed hybrid circuit + ridge",
}


def label(tag: str) -> str:
    base = tag.split("-")[0]
    if base in LABEL:
        return LABEL[base]
    if base == "matched_mlp":
        return "Param-matched MLP"
    parts = tag.split("-")
    if base == "fused":
        return "MLP + gated quantum fusion"
    return f"Quantum {parts[1]} / {parts[2]} / D={parts[3]} / {parts[4]}"


def s1_section(lines: list[str]) -> None:
    runs = load("s1_encoding")
    if not runs:
        return
    a = agg(runs)
    tags = sorted({t for t, _ in a})
    sizes = sorted({n for _, n in a})
    floor = runs[0]["noise_floor_mse"]
    lines += ["## S1 — amplitude–phase interaction (encoding comparison)", "",
              ("Target `y = A1·A2·cos(φ1 − φ2) + η`. Test MSE, mean ± std over 3 seeds; "
              f"target-noise floor = {floor:.4f}. Features are extracted by DFT from noisy series, "
              "so the floor is not reachable exactly."), "",
              "| model | qubits | 2q gates | meas. settings | " + " | ".join(f"n={n}" for n in sizes) + " |",
              "|---|---|---|---|" + "---|" * len(sizes)]
    order = sorted(tags, key=lambda t: a[(t, sizes[-1])]["mean"])
    for t in order:
        res = a[(t, sizes[0])]["runs"][0].get("resources", {})
        lines.append(f"| {label(t)} | {res.get('n_qubits', '–')} | {res.get('n_two_qubit_gates', '–')} | "
                     f"{res.get('measurement_settings', '–')} | "
                     + " | ".join(fmt(a[(t, n)]) for n in sizes) + " |")
    lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    left = ["raw_mlp", "fourier_mlp", "matched_mlp-hybrid-all_to_all-2-z", "explicit",
            "quantum-hybrid-all_to_all-2-z"]
    for i, t in enumerate([t for t in left if t in tags]):
        m = np.array([a[(t, n)]["mean"] for n in sizes])
        s = np.array([a[(t, n)]["std"] for n in sizes])
        axes[0].plot(sizes, m, marker="o", ms=6, color=SERIES[i], label=label(t))
        axes[0].fill_between(sizes, np.maximum(m - s, 1e-6), m + s, color=SERIES[i], alpha=0.12, lw=0)
    axes[0].set_title("Classical controls vs quantum (hybrid, Z readout)", loc="left")
    enc_color = {"angle": SERIES[0], "phase": SERIES[1], "hybrid": SERIES[2]}
    for t in tags:
        p = t.split("-")
        if p[0] != "quantum":
            continue
        m = [a[(t, n)]["mean"] for n in sizes]
        axes[1].plot(sizes, m, marker="o", ms=6, color=enc_color[p[1]],
                     ls="-" if p[4] == "z" else "--",
                     label=f"{p[1]}, " + ("Z readout + CZ" if p[4] == "z" else "Pauli-2 readout, no CZ"))
    axes[1].set_title("Quantum: encoding × readout", loc="left")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.axhline(floor, color=INK2, lw=1, ls=":")
        ax.text(sizes[0], floor * 1.15, "noise floor", color=INK2, fontsize=8)
        ax.set_xlabel("training samples")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("test MSE (log)")
    fig.tight_layout()
    fig.savefig(FIG / "s1_encoding.png", dpi=180)
    plt.close(fig)
    lines += ["![S1](figures/s1_encoding.png)", ""]
    _shots_table(lines, runs, sizes[-1])


def _shots_table(lines: list[str], runs: list[dict], n: int) -> None:
    rows = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["n_train"] == n and "shots_eval" in r:
            rows[r["tag"]]["exact"].append(r["test_mse"])
            for s, v in r["shots_eval"].items():
                rows[r["tag"]][s].append(v["mse"])
    if not rows:
        return
    cols = sorted({c for v in rows.values() for c in v if c != "exact"}, key=int)
    lines += [f"**Finite-shot evaluation** (trained at n={n}, 500 test samples; MSE mean over seeds)", "",
              "| model | exact | " + " | ".join(f"{c} shots" for c in cols) + " |",
              "|---|---|" + "---|" * len(cols)]
    for t, v in sorted(rows.items()):
        lines.append(f"| {label(t)} | {np.mean(v['exact']):.4f} | "
                     + " | ".join(f"{np.mean(v[c]):.4f}" for c in cols) + " |")
    lines.append("")


def s2_section(lines: list[str]) -> None:
    exps = [e for e in ("s2_topology_4q", "s2_topology_6q") if (OUT / e).exists() and load(e)]
    if not exps:
        return
    lines += ["## S2 — cross-period interaction (topology comparison)", "",
              ("One qubit per period patch, hybrid encoding, readout restricted to single-qubit ⟨Z⟩, "
              "so cross-patch terms must come from CZ entanglers. Parameter count is identical across "
              "topologies (fixed CZ). Test MSE, mean ± std over 3 seeds."), ""]
    fig, axes = plt.subplots(1, len(exps), figsize=(5.6 * len(exps), 4.2), sharey=True, squeeze=False)
    topo = ["none", "local", "cross_period", "all_to_all"]
    for ax, exp in zip(axes[0], exps):
        a = agg(load(exp))
        n = min({k[1] for k in a})
        nq = exp.split("_")[-1]
        lines += [f"### {nq}", "", "| model | depth | 2q gates | circuit depth | meas. settings | test MSE |",
                  "|---|---|---|---|---|---|"]
        for t in sorted({k[0] for k in a}, key=lambda t: a[(t, n)]["mean"]):
            res = a[(t, n)]["runs"][0].get("resources", {})
            d = t.split("-")[3] if t.startswith("quantum") else "–"
            lines.append(f"| {label(t)} | {d} | {res.get('n_two_qubit_gates', '–')} | {res.get('depth', '–')} | "
                         f"{res.get('measurement_settings', '–')} | {fmt(a[(t, n)])} |")
        lines.append("")
        x = np.arange(len(topo))
        for j, depth in enumerate((1, 2)):
            tags = [f"quantum-hybrid-{tp}-{depth}-z" for tp in topo]
            m = [a[(t, n)]["mean"] if (t, n) in a else np.nan for t in tags]
            s = [a[(t, n)]["std"] if (t, n) in a else 0 for t in tags]
            ax.bar(x + (j - 0.5) * 0.38, m, 0.36, yerr=s, color=SERIES[j], label=f"depth {depth}",
                   edgecolor=SURFACE, linewidth=2, error_kw={"ecolor": INK2, "lw": 1})
        for ref, ls in (("fourier_mlp", "--"), ("explicit", ":")):
            if (ref, n) in a:
                ax.axhline(a[(ref, n)]["mean"], color=INK2, lw=1, ls=ls)
                ax.text(len(topo) - 0.5, a[(ref, n)]["mean"] * 1.1, label(ref), color=INK2,
                        fontsize=7, ha="right")
        ax.set_xticks(x, ["none", "local", "cross-period", "all-to-all"])
        ax.set_yscale("log")
        ax.set_title(f"{nq}: entangling topology (Z readout)", loc="left")
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("test MSE (log)")
    fig.tight_layout()
    fig.savefig(FIG / "s2_topology.png", dpi=180)
    plt.close(fig)
    lines += ["![S2](figures/s2_topology.png)", ""]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    lines = ["# Module A results — Structured Quantum Periodic Interaction", "",
             ("Generated by `experiments/synthetic_periodic/make_plots.py` from the JSON run records in "
             "`outputs/module_a/` (simulator: PennyLane `default.qubit`, exact expectations unless noted)."), ""]
    s1_section(lines)
    s2_section(lines)
    (DOCS / "results_module_a.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
