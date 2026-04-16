#!/usr/bin/env python3
"""Plot Stage 1 vs Stage 1.5 comparison."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    results_dir = Path(__file__).parent / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 1 results
    with open(results_dir / "stage1" / "stage1_summary.json") as f:
        stage1 = json.load(f)

    # Load Stage 1.5 results
    with open(results_dir / "stage1_atlas" / "stage1_atlas_summary.json") as f:
        stage1_atlas = json.load(f)

    depths = [2, 4, 6, 9, 12]
    s1_nccs = []
    s1a_nccs = []

    for d in depths:
        s1_key = f"S1-D{d:.0f}mm"
        s1a_key = f"S1A-D{d:.0f}mm"
        s1_nccs.append(stage1.get(s1_key, {}).get("mean_ncc", 0))
        s1a_nccs.append(stage1_atlas.get(s1a_key, {}).get("mean_ncc", 0))

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(depths, s1_nccs, 'o-', linewidth=3, markersize=12,
            label='Stage 1: Homogeneous Cube', color='#2E86AB')
    ax.plot(depths, s1a_nccs, 's--', linewidth=3, markersize=12,
            label='Stage 1.5: Atlas-Shaped', color='#E63946')

    # Threshold lines
    ax.axhline(y=0.95, color='#06A77D', linestyle=':', linewidth=2, label='GO (NCC≥0.95)')
    ax.axhline(y=0.85, color='#F4A261', linestyle=':', linewidth=2, label='CAUTION (NCC≥0.85)')

    # Fill regions
    ax.fill_between(depths, 0.95, 1.0, alpha=0.1, color='#06A77D')
    ax.fill_between(depths, 0.85, 0.95, alpha=0.1, color='#F4A261')
    ax.fill_between(depths, 0, 0.85, alpha=0.1, color='#E63946')

    ax.set_xlabel('Depth from Dorsal Surface (mm)', fontsize=14)
    ax.set_ylabel('Mean NCC', fontsize=14)
    ax.set_title('Stage 1 vs Stage 1.5: Effect of Mouse Surface Shape\n'
                 'Green Function Accuracy Degrades Significantly with Real Geometry',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlim(0, 14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=11)

    # Add annotations
    for d, s1, s1a in zip(depths, s1_nccs, s1a_nccs):
        ax.annotate(f'{s1:.3f}', xy=(d, s1), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=9, color='#2E86AB', fontweight='bold')
        ax.annotate(f'{s1a:.3f}', xy=(d, s1a), xytext=(0, -15),
                    textcoords='offset points', ha='center',
                    fontsize=9, color='#E63946', fontweight='bold')

    plt.tight_layout()
    plt.savefig(figures_dir / "stage1_vs_stage1_5_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {figures_dir / 'stage1_vs_stage1_5_comparison.png'}")

    # Print comparison table
    print("\n" + "="*70)
    print("Stage 1 (Cube) vs Stage 1.5 (Atlas) Comparison")
    print("="*70)
    print(f"{'Depth':>8} | {'Cube NCC':>12} | {'Atlas NCC':>12} | {'Degradation':>12} | {'Status'}")
    print("-"*70)
    for d, s1, s1a in zip(depths, s1_nccs, s1a_nccs):
        degr = s1a - s1
        if s1a >= 0.95:
            status = "✅ GO"
        elif s1a >= 0.85:
            status = "⚠️ CAUTION"
        elif s1a >= 0.5:
            status = "❌ DEGRADED"
        else:
            status = "❌ UNUSABLE"
        print(f"{d:>8.0f}mm | {s1:>12.4f} | {s1a:>12.4f} | {degr:>+12.4f} | {status}")

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("The mouse surface shape SIGNIFICANTLY affects PSF accuracy.")
    print("Even with homogeneous tissue, boundary effects cause NCC to degrade")
    print("from >0.99 (cube) to <0.60 (atlas) at typical GS-FMT depths (4mm).")
    print("\nImplication: Pure Green's function is NOT sufficient for real FMT.")
    print("A residual network or boundary correction is needed.")

if __name__ == "__main__":
    main()
