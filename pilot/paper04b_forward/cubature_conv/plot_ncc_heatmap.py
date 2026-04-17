"""Plot NCC heatmaps for cubature convergence study."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_ball_heatmap(results_path: Path = None, output_path: Path = None):
    """Plot ball source NCC heatmap.

    Figure 4-A-1: scheme × r/δ
    """
    if results_path is None:
        results_path = Path(__file__).parent / "results" / "table_ball_RxN.json"
    if output_path is None:
        output_path = Path(__file__).parent / "results" / "fig_ball_heatmap.png"

    data = load_results(results_path)
    table = data["table"]

    schemes = ["SR6_7pt", "grid_27pt", "strat_33pt", "lebedev_50x4", "halton_64pt"]
    r_over_delta_values = sorted(set(row["r_over_delta"] for row in table))

    ncc_matrix = np.zeros((len(schemes), len(r_over_delta_values)))

    for row in table:
        r_over_delta = row["r_over_delta"]
        j = r_over_delta_values.index(r_over_delta)

        for i, scheme in enumerate(schemes):
            key = f"{scheme}_rel_l1"
            if key in row and row[key] is not None:
                rel_l1 = row[key]
                ncc = 1.0 - rel_l1
                ncc_matrix[i, j] = max(ncc_matrix[i, j], ncc)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(ncc_matrix, aspect="auto", cmap="RdYlGn", vmin=0.9, vmax=1.0)

    ax.set_xticks(range(len(r_over_delta_values)))
    ax.set_xticklabels([f"{v:.2f}" for v in r_over_delta_values])
    ax.set_xlabel("r/δ")

    ax.set_yticks(range(len(schemes)))
    ax.set_yticklabels(schemes)
    ax.set_ylabel("Cubature Scheme")

    ax.set_title("Ball Source: Cubature Accuracy (1 - Rel L1)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_gaussian_heatmap(results_path: Path = None, output_path: Path = None):
    """Plot Gaussian source NCC heatmap."""
    if results_path is None:
        results_path = Path(__file__).parent / "results" / "table_gaussian_sigmaxN.json"
    if output_path is None:
        output_path = Path(__file__).parent / "results" / "fig_gaussian_heatmap.png"

    data = load_results(results_path)
    table = data["table"]

    schemes = ["SR6_7pt", "grid_27pt", "strat_33pt", "lebedev_50x4", "halton_64pt"]
    sigma_over_delta_values = sorted(set(row["sigma_over_delta"] for row in table))

    ncc_matrix = np.zeros((len(schemes), len(sigma_over_delta_values)))

    for row in table:
        sigma_over_delta = row["sigma_over_delta"]
        j = sigma_over_delta_values.index(sigma_over_delta)

        for i, scheme in enumerate(schemes):
            key = f"{scheme}_rel_l1"
            if key in row and row[key] is not None:
                rel_l1 = row[key]
                ncc = 1.0 - rel_l1
                ncc_matrix[i, j] = max(ncc_matrix[i, j], ncc)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(ncc_matrix, aspect="auto", cmap="RdYlGn", vmin=0.9, vmax=1.0)

    ax.set_xticks(range(len(sigma_over_delta_values)))
    ax.set_xticklabels([f"{v:.2f}" for v in sigma_over_delta_values])
    ax.set_xlabel("σ/δ")

    ax.set_yticks(range(len(schemes)))
    ax.set_yticklabels(schemes)
    ax.set_ylabel("Cubature Scheme")

    ax.set_title("Gaussian Source: Cubature Accuracy (1 - Rel L1)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_critical_r_over_delta(results_path: Path = None, output_path: Path = None):
    """Plot critical r/δ for each scheme to reach NCC ≥ 0.99.

    Figure 4-A-2
    """
    if results_path is None:
        results_path = Path(__file__).parent / "results" / "table_ball_RxN.json"
    if output_path is None:
        output_path = (
            Path(__file__).parent / "results" / "fig_critical_r_over_delta.png"
        )

    data = load_results(results_path)
    table = data["table"]

    schemes = ["SR6_7pt", "grid_27pt", "strat_33pt", "lebedev_50x4", "halton_64pt"]
    threshold = 0.99

    critical_values = {}
    for scheme in schemes:
        max_r = 0
        for row in table:
            key = f"{scheme}_rel_l1"
            if key in row and row[key] is not None:
                accuracy = 1.0 - row[key]
                if accuracy >= threshold:
                    max_r = max(max_r, row["r_over_delta"])
        critical_values[scheme] = max_r if max_r > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(schemes))
    y = [critical_values[s] for s in schemes]

    bars = ax.bar(x, y, color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, rotation=15)
    ax.set_ylabel("Critical r/δ (NCC ≥ 0.99)")
    ax.set_title("Maximum r/δ Achieving NCC ≥ 0.99")

    for bar, val in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, max(y) * 1.2 if max(y) > 0 else 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    plot_ball_heatmap()
    plot_gaussian_heatmap()
    plot_critical_r_over_delta()


if __name__ == "__main__":
    main()
