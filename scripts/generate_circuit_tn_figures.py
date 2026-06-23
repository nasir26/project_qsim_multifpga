"""
generate_circuit_tn_figures.py

Generates publication-quality figures for the three benchmark circuit families
(GHZ, 1Q-Rots, Brick-wall) showing:
  1. The quantum circuit diagram with gate labels and depth annotation
  2. The corresponding MPS tensor-network diagram

All figures are written to  ../paper/figs/  relative to this script.
Run from the qsim_multifpga root or directly:
    python scripts/generate_circuit_tn_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

# ── output directory ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR   = os.path.join(SCRIPT_DIR, "..", "paper", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

# ── colour palette (colour-blind friendly) ────────────────────────────────────
C_H      = "#4477AA"   # Hadamard – blue
C_CX     = "#EE6677"   # CNOT – red
C_RY     = "#228833"   # RY   – green
C_CTRL   = "#CCBB44"   # control dot – gold
C_WIRE   = "#333333"
C_BG     = "#FFFFFF"
C_TENSOR = "#66CCEE"   # MPS tensor – cyan
C_BOND   = "#AA3377"   # bond leg   – purple
C_PHY    = "#228833"   # physical leg – green
C_TRUNC  = "#EE6677"   # truncated bond


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – Quantum circuit figures (Qiskit mpl backend)
# ═══════════════════════════════════════════════════════════════════════════════

def make_ghz_circuit(n: int = 6) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"GHZ ({n} qubits)")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def make_1qrots_circuit(n: int = 6) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"1Q-Rots ({n} qubits)")
    angles = np.linspace(np.pi / 6, 5 * np.pi / 6, n)
    for i, theta in enumerate(angles):
        qc.ry(theta, i)
    qc.measure_all()
    return qc


def make_brickwall_circuit(n: int = 6, layers: int = 3) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"Brick-wall ({n}q, {layers} CNOT layers)")
    angles = np.linspace(np.pi / 4, 3 * np.pi / 4, n)
    for i, theta in enumerate(angles):
        qc.ry(theta, i)
    for layer in range(layers):
        offset = layer % 2
        for i in range(offset, n - 1, 2):
            qc.cx(i, i + 1)
        angles2 = np.linspace(np.pi / 3, 2 * np.pi / 3, n)
        for i, theta in enumerate(angles2):
            qc.ry(theta, i)
    qc.measure_all()
    return qc


def save_circuit_figure(qc: QuantumCircuit, fname: str, title: str) -> None:
    ops   = dict(qc.count_ops())
    depth = qc.depth()

    fig = qc.draw(
        output="mpl",
        style={
            "backgroundcolor": C_BG,
            "linecolor":       C_WIRE,
            "gatefacecolor":   C_H,
            "textcolor":       "black",
            "fontsize":        11,
        },
        fold=25,
        plot_barriers=False,
    )

    # annotation box
    gate_str = "  ".join(f"{g}: {cnt}" for g, cnt in sorted(ops.items()))
    ann = f"Gates — {gate_str}\nCircuit depth: {depth}"
    fig.text(
        0.01, -0.04, ann,
        ha="left", va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4FF", edgecolor="#AABBCC"),
        transform=fig.transFigure,
    )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    out = os.path.join(FIGS_DIR, fname)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – Summary table figure (gates + depth for all three families)
# ═══════════════════════════════════════════════════════════════════════════════

def make_circuit_summary_figure() -> None:
    """3-panel figure: one circuit per family, with gate/depth info."""
    circuits = [
        (make_ghz_circuit(5),          "GHZ\n(5 qubits, depth 6)",     "fig_5a_circuit_ghz.png"),
        (make_1qrots_circuit(5),        "1Q-Rots\n(5 qubits, depth 2)", "fig_5b_circuit_1qrots.png"),
        (make_brickwall_circuit(6, 2),  "Brick-wall\n(6 qubits, 2 CNOT layers)", "fig_5c_circuit_brickwall.png"),
    ]
    for qc, title, fname in circuits:
        save_circuit_figure(qc, fname, title)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – MPS tensor-network diagrams
# ═══════════════════════════════════════════════════════════════════════════════

def draw_mps_chain(
    ax, n: int, chi_labels, phy_labels=None,
    highlight_sites=None,      # sites to colour differently (e.g. during contraction)
    show_theta=False,          # draw the merged Θ tensor at sites highlight_sites
    title: str = "",
    bond_dim_note: str = "",
):
    """
    Draw an MPS chain on *ax*.

    Parameters
    ----------
    n            : number of sites
    chi_labels   : list of n+1 bond-dimension labels (chi_0 … chi_n)
    phy_labels   : list of n physical-index labels (default: σ_0 … σ_{n-1})
    highlight_sites: list of site indices to colour as active (contraction pair)
    show_theta   : if True, merge highlight_sites[0] and [1] into a single Θ box
    """
    if phy_labels is None:
        phy_labels = [rf"$\sigma_{{{i}}}$" for i in range(n)]

    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(-1.8, 1.4)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    xs = list(range(n))
    y0 = 0.0
    box_w, box_h = 0.55, 0.55

    highlight_sites = set(highlight_sites or [])

    # ── bond wires ──────────────────────────────────────────────────────────
    for i in range(n - 1):
        x_start = xs[i]   + box_w / 2
        x_end   = xs[i+1] - box_w / 2
        color   = C_TRUNC if (i in highlight_sites and i+1 in highlight_sites) else C_BOND
        ax.plot([x_start, x_end], [y0, y0],
                color=color, lw=2.5, zorder=1)
        # bond-dimension label
        mid_x = (x_start + x_end) / 2
        ax.text(mid_x, y0 + 0.22, chi_labels[i + 1],
                ha="center", va="bottom", fontsize=8.5, color=C_BOND,
                fontweight="bold")

    # boundary stubs
    ax.plot([xs[0] - box_w / 2 - 0.25, xs[0] - box_w / 2], [y0, y0],
            color=C_BOND, lw=2.5, zorder=1)
    ax.plot([xs[-1] + box_w / 2, xs[-1] + box_w / 2 + 0.25], [y0, y0],
            color=C_BOND, lw=2.5, zorder=1)
    ax.text(xs[0] - box_w / 2 - 0.3, y0, chi_labels[0],
            ha="right", va="center", fontsize=8, color=C_BOND, fontweight="bold")
    ax.text(xs[-1] + box_w / 2 + 0.3, y0, chi_labels[-1],
            ha="left", va="center", fontsize=8, color=C_BOND, fontweight="bold")

    # ── physical legs ────────────────────────────────────────────────────────
    for i in range(n):
        ax.annotate(
            "", xy=(xs[i], y0 - box_h / 2 - 0.55),
            xytext=(xs[i], y0 - box_h / 2),
            arrowprops=dict(arrowstyle="-|>", color=C_PHY, lw=1.6),
        )
        ax.text(xs[i], y0 - box_h / 2 - 0.72,
                phy_labels[i],
                ha="center", va="top", fontsize=9, color=C_PHY)

    # ── site tensors ─────────────────────────────────────────────────────────
    if show_theta and len(highlight_sites) >= 2:
        hs = sorted(highlight_sites)[:2]
        # draw merged Θ spanning hs[0]..hs[1]
        x_left  = xs[hs[0]]  - box_w / 2
        x_right = xs[hs[1]]  + box_w / 2
        theta_box = FancyBboxPatch(
            (x_left, y0 - box_h / 2),
            x_right - x_left, box_h,
            boxstyle="round,pad=0.05",
            facecolor=C_TRUNC, edgecolor="black", lw=1.8, zorder=3,
        )
        ax.add_patch(theta_box)
        mid = (xs[hs[0]] + xs[hs[1]]) / 2
        ax.text(mid, y0, r"$\Theta$", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white", zorder=4)
        # draw non-highlighted sites normally
        for i in range(n):
            if i in hs:
                continue
            _draw_tensor_box(ax, xs[i], y0, box_w, box_h, i, highlight=False)
    else:
        for i in range(n):
            _draw_tensor_box(ax, xs[i], y0, box_w, box_h, i,
                             highlight=(i in highlight_sites))

    if bond_dim_note:
        ax.text(0.5, -0.07, bond_dim_note, transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5,
                style="italic", color="#555555")


def _draw_tensor_box(ax, x, y, w, h, idx, highlight=False):
    fc = C_TENSOR if not highlight else "#FF9900"
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=fc, edgecolor="black", lw=1.6, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, rf"$A^{{({idx})}}$",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color="black", zorder=4)


# ── per-family tensor-network diagrams ───────────────────────────────────────

def make_tn_ghz(n: int = 6) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    fig.suptitle(f"GHZ circuit ({n} qubits) — MPS tensor-network view",
                 fontsize=12, fontweight="bold", y=1.04)

    # left: initial product state (chi=1 everywhere)
    chi_prod = [r"$\chi{=}1$"] * (n + 1)
    draw_mps_chain(
        axes[0], n, chi_prod,
        title="(a) Initial product state  $|0\\rangle^{\\otimes n}$",
        bond_dim_note="Bond dim = 1 everywhere (product state)",
    )

    # right: after GHZ prep (chi=2 across all cuts)
    chi_ghz = [r"$\chi{=}1$"] + [r"$\chi{=}2$"] * (n - 1) + [r"$\chi{=}1$"]
    draw_mps_chain(
        axes[1], n, chi_ghz,
        title="(b) After GHZ preparation  $|\\mathrm{GHZ}\\rangle$",
        bond_dim_note=r"Bond dim = 2 (saturated after first CNOT layer)",
    )

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "fig_6a_tn_ghz.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def make_tn_1qrots(n: int = 6) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.0))
    fig.suptitle(f"1Q-Rots circuit ({n} qubits) — MPS tensor-network view",
                 fontsize=12, fontweight="bold", y=1.04)

    chi_sep = [r"$\chi{=}1$"] * (n + 1)
    phy = [rf"$R_y(\theta_{{{i}}})$" for i in range(n)]
    draw_mps_chain(
        ax, n, chi_sep, phy_labels=phy,
        title="After $R_y$ rotations — no entanglement generated",
        bond_dim_note=r"Bond dim = 1 throughout (separable state, no two-qubit gates)",
    )

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "fig_6b_tn_1qrots.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def make_tn_brickwall(n: int = 6, layers: int = 2) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.4))
    fig.suptitle(
        f"Brick-wall circuit ({n} qubits, {layers} CNOT layers) — MPS tensor-network view",
        fontsize=12, fontweight="bold", y=1.04,
    )

    # panel (a): initial
    chi0 = [r"$\chi{=}1$"] * (n + 1)
    draw_mps_chain(axes[0], n, chi0,
                   title="(a) Initial product state",
                   bond_dim_note="χ = 1 everywhere")

    # panel (b): mid-circuit two-site contraction (Θ on sites 2-3)
    chi_mid = [r"$\chi{=}1$", r"$\chi{=}2$", r"$\chi{=}2$",
               r"$\chi{=}4$", r"$\chi{=}2$", r"$\chi{=}2$", r"$\chi{=}1$"]
    if len(chi_mid) != n + 1:
        chi_mid = [r"$\chi{=}1$"] + [r"$\chi{=}\chi$"] * (n - 1) + [r"$\chi{=}1$"]
    draw_mps_chain(axes[1], n, chi_mid,
                   highlight_sites=[2, 3], show_theta=True,
                   title=r"(b) During gate: $\Theta = A^{(2)} \cdot A^{(3)}$ then SVD",
                   bond_dim_note="Joined tensor Θ (sites 2-3) before SVD truncation")

    # panel (c): after full circuit (bond dim grown and truncated to chi_max)
    chi_full = [r"$\chi{=}1$"] + [r"$\chi{\leq}\chi_{\max}$"] * (n - 1) + [r"$\chi{=}1$"]
    draw_mps_chain(axes[2], n, chi_full,
                   title=r"(c) After circuit — bonds truncated to $\chi_{\max}$",
                   bond_dim_note=r"$\chi_{\max} = 64$ in benchmark runs")

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "fig_6c_tn_brickwall.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – Combined circuit + TN panels (one figure per family)
# ═══════════════════════════════════════════════════════════════════════════════

def make_combined_figure(
    qc: QuantumCircuit,
    tn_fn,           # callable that draws TN on a given axes
    fname: str,
    suptitle: str,
) -> None:
    """Build a 2-row figure: top = circuit, bottom = tensor network."""
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    # ── top row: circuit (drawn via Qiskit, then embedded) ──────────────────
    qc_fig = qc.draw(output="mpl",
                     style={"backgroundcolor": C_BG, "fontsize": 10},
                     fold=30, plot_barriers=False)
    qc_fig.canvas.draw()
    buf = qc_fig.canvas.buffer_rgba()
    import numpy as np
    circuit_img = np.frombuffer(buf, dtype=np.uint8)
    w_px, h_px = qc_fig.canvas.get_width_height()
    circuit_img = circuit_img.reshape(h_px, w_px, 4)
    plt.close(qc_fig)

    ax_circ = fig.add_axes([0.02, 0.52, 0.96, 0.46])
    ax_circ.imshow(circuit_img)
    ax_circ.axis("off")

    ops   = dict(qc.count_ops())
    depth = qc.depth()
    gate_str = "  |  ".join(f"{g}: {cnt}" for g, cnt in sorted(ops.items()))
    ax_circ.set_title(
        f"Circuit — Gates: {gate_str}   |   Depth: {depth}",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4FF", edgecolor="#AABBCC"),
        pad=4,
    )

    # ── bottom row: tensor network ───────────────────────────────────────────
    ax_tn = fig.add_axes([0.02, 0.02, 0.96, 0.44])
    tn_fn(ax_tn)

    out = os.path.join(FIGS_DIR, fname)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – All-in-one summary figure (3 columns × 2 rows)
# ═══════════════════════════════════════════════════════════════════════════════

def make_master_figure() -> None:
    """
    3-column × 2-row figure.
    Top row: circuit diagrams (embedded Qiskit images).
    Bottom row: MPS tensor-network diagrams.
    """
    n_ghz, n_rots, n_bw = 5, 5, 6
    bw_layers = 2

    circuits = [
        make_ghz_circuit(n_ghz),
        make_1qrots_circuit(n_rots),
        make_brickwall_circuit(n_bw, bw_layers),
    ]
    col_titles = [
        f"GHZ  ({n_ghz} qubits)",
        f"1Q-Rots  ({n_rots} qubits)",
        f"Brick-wall  ({n_bw}q, {bw_layers} CNOT layers)",
    ]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Benchmark circuit families: quantum circuits (top) and "
        "corresponding MPS tensor networks (bottom)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── top row: circuits ────────────────────────────────────────────────────
    for col, (qc, ctitle) in enumerate(zip(circuits, col_titles)):
        qc_fig = qc.draw(output="mpl",
                         style={"backgroundcolor": C_BG, "fontsize": 9},
                         fold=20, plot_barriers=False)
        qc_fig.canvas.draw()
        buf = qc_fig.canvas.buffer_rgba()
        import numpy as np
        img = np.frombuffer(buf, dtype=np.uint8)
        w_px, h_px = qc_fig.canvas.get_width_height()
        img = img.reshape(h_px, w_px, 4)
        plt.close(qc_fig)

        left   = 0.03 + col * 0.335
        ax_c   = fig.add_axes([left, 0.53, 0.30, 0.42])
        ax_c.imshow(img)
        ax_c.axis("off")

        ops   = dict(qc.count_ops())
        depth = qc.depth()
        gate_str = ", ".join(f"{g}:{v}" for g, v in sorted(ops.items()))
        ax_c.set_title(
            f"{ctitle}\nGates — {gate_str}   depth={depth}",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0F4FF",
                      edgecolor="#AABBCC"),
            pad=3,
        )

    # ── bottom row: tensor networks ──────────────────────────────────────────
    # GHZ TN
    ax_tn0 = fig.add_axes([0.03, 0.04, 0.29, 0.42])
    chi_ghz = ([r"1"] + [r"2"] * (n_ghz - 1) + [r"1"])
    chi_ghz_lab = [rf"$\chi{{\!=\!}}{c}$" for c in chi_ghz]
    draw_mps_chain(ax_tn0, n_ghz, chi_ghz_lab,
                   title="GHZ MPS  (χ = 2 after CNOT chain)",
                   bond_dim_note="χ saturates at 2 — GHZ is maximally\nentangled but only needs bond dim 2")

    # 1Q-Rots TN
    ax_tn1 = fig.add_axes([0.365, 0.04, 0.29, 0.42])
    chi_sep = [r"$\chi{=}1$"] * (n_rots + 1)
    draw_mps_chain(ax_tn1, n_rots, chi_sep,
                   title="1Q-Rots MPS  (χ = 1 throughout)",
                   bond_dim_note="No entanglement — product state,\nbond dim stays at 1")

    # Brick-wall TN
    ax_tn2 = fig.add_axes([0.695, 0.04, 0.29, 0.42])
    chi_bw = ([r"$\chi{=}1$"]
              + [r"$\chi{\leq}\chi_{\max}$"] * (n_bw - 1)
              + [r"$\chi{=}1$"])
    draw_mps_chain(ax_tn2, n_bw, chi_bw,
                   highlight_sites=[2, 3], show_theta=True,
                   title=r"Brick-wall MPS  ($\chi \leq \chi_{\max}$)",
                   bond_dim_note=r"Bond dim grows with depth; truncated to $\chi_{\max}=64$")

    out = os.path.join(FIGS_DIR, "fig_5_circuits_and_tn.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – Bond dimension growth diagram
# ═══════════════════════════════════════════════════════════════════════════════

def make_bond_growth_figure() -> None:
    """
    Shows how the bond dimension profile evolves across the chain as entanglement
    is injected by CNOT layers in the brick-wall family.
    """
    n       = 8
    n_steps = 4   # number of CNOT layers shown
    chi_max = 8

    def chi_profile_after_layer(layer: int) -> list:
        """Approximate bond-dim profile after `layer` CNOT layers."""
        profile = [1] * (n + 1)
        for step in range(layer):
            offset = step % 2
            for i in range(offset, n - 1, 2):
                profile[i + 1] = min(chi_max, profile[i + 1] * 2 + 1)
        profile[0] = 1
        profile[-1] = 1
        return profile

    fig, axes = plt.subplots(1, n_steps + 1, figsize=(18, 3.2), sharey=True)
    fig.suptitle(
        "Bond-dimension profile growth — Brick-wall family (8 qubits)",
        fontsize=12, fontweight="bold",
    )

    xs = np.arange(n + 1)  # bond positions 0..n

    for step in range(n_steps + 1):
        ax  = axes[step]
        chi = chi_profile_after_layer(step)
        ax.bar(xs, chi, color=C_BOND, alpha=0.7, width=0.6, edgecolor="black", lw=0.8)
        ax.axhline(chi_max, color=C_TRUNC, ls="--", lw=1.4, label=r"$\chi_{\max}$")
        ax.set_xticks(xs)
        ax.set_xticklabels([rf"$\chi_{{{i}}}$" for i in range(n + 1)],
                           fontsize=7.5, rotation=45)
        ax.set_ylim(0, chi_max + 2)
        ax.set_yticks([1, 2, 4, 8])
        ax.set_title(f"Layer {step}" if step > 0 else "Initial",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Bond index", fontsize=8)
        if step == 0:
            ax.set_ylabel("Bond dimension χ", fontsize=9)

    axes[-1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "fig_7_bond_growth.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("Generating circuit + tensor-network figures …")

    print("\n[1/5] Individual circuit diagrams …")
    make_circuit_summary_figure()

    print("\n[2/5] Individual tensor-network diagrams …")
    make_tn_ghz(n=6)
    make_tn_1qrots(n=6)
    make_tn_brickwall(n=6, layers=2)

    print("\n[3/5] Master combined figure (circuits + tensor networks) …")
    make_master_figure()

    print("\n[4/5] Bond-dimension growth figure …")
    make_bond_growth_figure()

    print("\n[5/5] Done.  Figures written to:")
    for f in sorted(os.listdir(FIGS_DIR)):
        if f.startswith("fig_5") or f.startswith("fig_6") or f.startswith("fig_7"):
            print(f"   {os.path.join(FIGS_DIR, f)}")
