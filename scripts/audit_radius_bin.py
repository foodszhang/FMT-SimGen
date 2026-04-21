#!/usr/bin/env python3
"""
S3: Radius Bin Validation — verify representative vs upper-bound acceptance rates.

Tests whether using representative radius (2.25/2.75/3.25) vs
upper-bound radius (2.5/3.0/3.5) produces different validation outcomes.
Writes: analysis/radius_bin_validation_report.json
"""
import numpy as np
import sys, json, pathlib
sys.path.insert(0, '.')

from fmt_simgen.atlas.digimouse import DigimouseAtlas
from fmt_simgen.tumor.tumor_generator import TumorGenerator
from fmt_simgen.frame_contract import TRUNK_OFFSET_ATLAS_MM

def run():
    atlas_path = "/home/foods/pro/mcx_simulation/ct_data/atlas_380x992x208.hdr"
    atlas = DigimouseAtlas(atlas_path)
    atlas.load()

    vol_raw = np.fromfile('output/shared/mcx_volume_trunk.bin', dtype=np.uint8)
    vol_xyz = vol_raw.reshape([104, 200, 190]).transpose(2, 1, 0)

    tg = TumorGenerator(
        config={"regions": ["dorsal", "lateral"],
                "num_foci_distribution": {1: 1.0},
                "shapes": ["sphere"],
                "radius_range": [2.0, 3.5],
                "depth_range": [2.0, 8.5],
                "depth_distribution": {"shallow": {"range": [2.0, 4.0]}},
                "max_cluster_radius": 8.0,
                "min_foci_distance_abs": 3.0,
                "ellipsoid_axis_ratio": [1.2, 1.5]},
        atlas=atlas,
        merged_voxel_volume=vol_xyz,
        voxel_size_mm=0.2,
        trunk_offset_mm=TRUNK_OFFSET_ATLAS_MM.copy(),
    )

    radius_bins = [
        ("small",  2.0, 2.5, 2.25, 2.5),
        ("medium", 2.5, 3.0, 2.75, 3.0),
        ("large",  3.0, 3.5, 3.25, 3.5),
    ]

    results = {}
    for bin_name, r_min, r_max, r_rep, r_upper in radius_bins:
        all_positions = []
        for zone_name, pool in tg._safe_zone_pools.items():
            for pos in pool:
                all_positions.append((zone_name, pos))

        rng = np.random.default_rng(42)
        n_sample = min(500, len(all_positions))
        indices = rng.choice(len(all_positions), n_sample, replace=False)
        sample_positions = [all_positions[i] for i in indices]

        rep_valid = rep_fails_upper = upper_valid = 0
        for zone_name, pos in sample_positions:
            rep_ok, _ = tg.is_valid_placement(pos, r_rep, record_rejections=False)
            upper_ok, _ = tg.is_valid_placement(pos, r_upper, record_rejections=False)
            if rep_ok:
                rep_valid += 1
                if not upper_ok:
                    rep_fails_upper += 1
            if upper_ok:
                upper_valid += 1

        n = len(sample_positions)
        false_safe_rate = rep_fails_upper / max(rep_valid, 1)
        results[bin_name] = {
            "range_mm": [r_min, r_max],
            "rep_radius_mm": r_rep,
            "upper_radius_mm": r_upper,
            "pool_total": len(all_positions),
            "sampled_n": n,
            "rep_valid": rep_valid,
            "rep_valid_pct": round(rep_valid / n * 100, 1),
            "upper_valid": upper_valid,
            "upper_valid_pct": round(upper_valid / n * 100, 1),
            "false_safe_rate": round(false_safe_rate, 4),
            "false_safe_n": rep_fails_upper,
            "gate_pass": false_safe_rate < 0.05,
        }
        print(f"{bin_name}: rep={rep_valid}/{n} ({rep_valid/n*100:.1f}%)  "
              f"upper={upper_valid}/{n} ({upper_valid/n*100:.1f}%)  "
              f"false_safe={rep_fails_upper}/{rep_valid} ({false_safe_rate*100:.1f}%)")

    out = {"radius_bins": results,
           "recommendation": "use_upper_bound" if any(r["false_safe_rate"] > 0.01 for r in results.values()) else "use_representative"}
    pathlib.Path("analysis").mkdir(exist_ok=True)
    pathlib.Path("analysis/radius_bin_validation_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: analysis/radius_bin_validation_report.json")
    print(f"Recommendation: {out['recommendation']}")

if __name__ == "__main__":
    run()