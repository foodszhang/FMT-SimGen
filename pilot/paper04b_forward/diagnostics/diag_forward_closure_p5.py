"""Forward Closure Round 1 — P5-ventral residual attribution."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf
from shared.metrics import scale_factor_logmse

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
ARCHIVE_BASE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")

SOFT_TISSUE_LABEL = 1
AIR_LABEL = 0
LIVER_LABEL = 6
BONE_LABEL = 7

GT_POS = np.array([-0.6, 2.4, -3.8])
EPS = 1e-20


def load_volume():
    return np.fromfile(VOLUME_PATH, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)


def extract_surface_vertices(binary_mask, voxel_size_mm):
    verts, faces, normals, _ = measure.marching_cubes(
        binary_mask.astype(float), level=0.5, spacing=(voxel_size_mm,) * 3
    )
    center = np.array(binary_mask.shape) / 2 * voxel_size_mm
    verts_physical = verts - center
    return verts_physical, faces, normals


def is_direct_path_vertex(source_pos_mm, vertex_pos_mm, volume_labels, voxel_size_mm):
    center = np.array(volume_labels.shape) / 2

    def mm_to_voxel(mm):
        return np.floor(mm / voxel_size_mm + center).astype(int)

    direction = vertex_pos_mm - source_pos_mm
    distance = np.linalg.norm(direction)
    if distance < 0.01:
        return True
    direction = direction / distance
    step_mm = 0.1
    n_steps = int(distance / step_mm)

    for i in range(1, n_steps + 1):
        pos_mm = source_pos_mm + i * step_mm * direction
        voxel = mm_to_voxel(pos_mm)
        if not (
            0 <= voxel[0] < volume_labels.shape[0]
            and 0 <= voxel[1] < volume_labels.shape[1]
            and 0 <= voxel[2] < volume_labels.shape[2]
        ):
            break
        label = volume_labels[voxel[0], voxel[1], voxel[2]]
        if label not in {AIR_LABEL, SOFT_TISSUE_LABEL}:
            return False
    return True


def get_path_info(source_pos_mm, vertex_pos_mm, volume_labels, voxel_size_mm):
    center = np.array(volume_labels.shape) / 2

    def mm_to_voxel(mm):
        return np.floor(mm / voxel_size_mm + center).astype(int)

    direction = vertex_pos_mm - source_pos_mm
    distance = np.linalg.norm(direction)
    if distance < 0.01:
        return "soft-only", {SOFT_TISSUE_LABEL: 0.0}

    direction = direction / distance
    step_mm = 0.1
    n_steps = int(distance / step_mm)

    labels_encountered = set()
    first_organ = None
    path_lengths = {}

    for i in range(1, n_steps + 1):
        pos_mm = source_pos_mm + i * step_mm * direction
        voxel = mm_to_voxel(pos_mm)
        if not (
            0 <= voxel[0] < volume_labels.shape[0]
            and 0 <= voxel[1] < volume_labels.shape[1]
            and 0 <= voxel[2] < volume_labels.shape[2]
        ):
            break
        label = volume_labels[voxel[0], voxel[1], voxel[2]]
        labels_encountered.add(int(label))
        path_lengths[label] = path_lengths.get(label, 0) + step_mm
        if label not in {AIR_LABEL, SOFT_TISSUE_LABEL} and first_organ is None:
            first_organ = int(label)

    organ_labels = labels_encountered - {AIR_LABEL, SOFT_TISSUE_LABEL}
    if len(organ_labels) == 0:
        path_class = "soft-only"
    elif LIVER_LABEL in organ_labels:
        path_class = "through-liver"
    elif BONE_LABEL in organ_labels:
        path_class = "through-bone"
    else:
        path_class = "through-other-organs"

    return path_class, path_lengths, first_organ


def sample_fluence_at_vertices(fluence, vertices_mm, voxel_size_mm):
    center = np.array(fluence.shape) / 2
    verts_voxel = np.floor(vertices_mm / voxel_size_mm + center).astype(int)
    phi = np.zeros(len(vertices_mm), dtype=np.float32)
    valid = np.zeros(len(vertices_mm), dtype=bool)
    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if (
            0 <= vx < fluence.shape[0]
            and 0 <= vy < fluence.shape[1]
            and 0 <= vz < fluence.shape[2]
        ):
            phi[i] = fluence[vx, vy, vz]
            valid[i] = True
    return phi, valid


def compute_green_at_vertices(vertices_mm, source_pos_mm, optical):
    dx = vertices_mm[:, 0] - source_pos_mm[0]
    dy = vertices_mm[:, 1] - source_pos_mm[1]
    dz = vertices_mm[:, 2] - source_pos_mm[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.maximum(r, 0.01)
    phi_green = G_inf(r, optical).astype(np.float64)
    return phi_green, r


def compute_residual(measurement, green, scale, eps=1e-20):
    log_mcx = np.log10(measurement + eps)
    log_green = np.log10(scale * green + eps)
    residual = log_mcx - log_green
    return residual


def residual_stats(residual):
    return {
        "mean": float(np.mean(residual)),
        "median": float(np.median(residual)),
        "std": float(np.std(residual)),
        "p5": float(np.percentile(residual, 5)),
        "p25": float(np.percentile(residual, 25)),
        "p75": float(np.percentile(residual, 75)),
        "p95": float(np.percentile(residual, 95)),
        "max_abs": float(np.max(np.abs(residual))),
        "frac_lt_0.05": float(np.mean(np.abs(residual) < 0.05)),
        "frac_lt_0.1": float(np.mean(np.abs(residual) < 0.1)),
        "frac_lt_0.2": float(np.mean(np.abs(residual) < 0.2)),
    }


def save_residual_csv(data, output_path):
    fieldnames = [
        "vertex_idx",
        "x",
        "y",
        "z",
        "measurement",
        "green",
        "scale",
        "residual",
        "distance",
        "angle_cos",
        "path_class",
        "soft_len",
        "liver_len",
        "bone_len",
        "other_len",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in data:
            w.writerow(row)


def plot_histogram(residual, title, output_path):
    plt.figure(figsize=(8, 6))
    plt.hist(residual, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Residual: log10(MCX) - log10(k·Green)")
    plt.ylabel("Count")
    plt.title(title)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_vs_distance(residual, distance, title, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(distance, residual, alpha=0.3, s=1)
    plt.xlabel("Distance r (mm)")
    plt.ylabel("Residual")
    plt.title(title)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_vs_angle(residual, angle_cos, title, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(angle_cos, residual, alpha=0.3, s=1)
    plt.xlabel("cos(angle)")
    plt.ylabel("Residual")
    plt.title(title)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_by_pathclass(residuals_by_class, output_path):
    plt.figure(figsize=(10, 6))
    labels = list(residuals_by_class.keys())
    data = [residuals_by_class[k] for k in labels]
    plt.boxplot(data, labels=labels)
    plt.ylabel("Residual")
    plt.xlabel("Path class")
    plt.title("Residual by path class")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_html_surface(vertices, residual, title, output_path, source_pos=None):
    try:
        import plotly.graph_objects as go
        from skimage import measure

        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        fig = go.Figure(
            data=go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=2,
                    color=residual,
                    colorscale="RdBu",
                    colorbar=dict(title="Residual"),
                    cmin=-np.max(np.abs(residual)),
                    cmax=np.max(np.abs(residual)),
                ),
                text=[f"R={r:.3f}" for r in residual],
                hoverinfo="text",
            )
        )

        if source_pos is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[source_pos[0]],
                    y=[source_pos[1]],
                    z=[source_pos[2]],
                    mode="markers",
                    marker=dict(size=8, color="green", symbol="diamond"),
                    name="Source",
                )
            )

        fig.update_layout(
            title=title,
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        )
        fig.write_html(output_path)
    except ImportError:
        print("plotly not available, skipping HTML surface")


def main():
    print("=" * 70)
    print("Forward Closure Round 1 — P5-ventral residual attribution")
    print("=" * 70)

    output_dir = Path("pilot/paper04b_forward/results/forward_closure_p5")
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    binary_mask = volume > 0

    print("Extracting surface vertices...")
    vertices_mm, faces, normals = extract_surface_vertices(binary_mask, VOXEL_SIZE_MM)
    n_vertices = len(vertices_mm)
    print(f"Total vertices: {n_vertices}")

    fluence_path = ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy"
    print(f"Loading fluence from {fluence_path}")
    fluence = np.load(fluence_path)

    measurement, valid_in_bounds = sample_fluence_at_vertices(
        fluence, vertices_mm, VOXEL_SIZE_MM
    )
    print(f"Valid in bounds: {np.sum(valid_in_bounds)}")

    print("Computing direct-path mask...")
    direct_mask_gt = np.zeros(n_vertices, dtype=bool)
    for i in range(n_vertices):
        direct_mask_gt[i] = is_direct_path_vertex(
            GT_POS, vertices_mm[i], volume, VOXEL_SIZE_MM
        )
    print(f"Direct-path vertices: {np.sum(direct_mask_gt)}")

    print("Computing Green forward...")
    green, distances = compute_green_at_vertices(vertices_mm, GT_POS, OPTICAL)

    print("Computing path info...")
    path_classes = []
    path_lengths_list = []
    first_organs = []
    for i in range(n_vertices):
        pc, pl, fo = get_path_info(GT_POS, vertices_mm[i], volume, VOXEL_SIZE_MM)
        path_classes.append(pc)
        path_lengths_list.append(pl)
        first_organs.append(fo)

    dx = vertices_mm[:, 0] - GT_POS[0]
    dy = vertices_mm[:, 1] - GT_POS[1]
    dz = vertices_mm[:, 2] - GT_POS[2]
    direction = np.stack([dx, dy, dz], axis=1)
    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction_unit = direction / np.maximum(direction_norm, 0.01)

    if len(normals) == n_vertices:
        angle_cos = np.abs(np.sum(normals * direction_unit, axis=1))
    else:
        angle_cos = np.ones(n_vertices)

    mask_direct = valid_in_bounds & direct_mask_gt & (measurement > 1e-10)
    mask_all = valid_in_bounds & (measurement > 1e-10)

    print(f"\nMask A (direct-path): {np.sum(mask_direct)} vertices")
    print(f"Mask B (all-surface): {np.sum(mask_all)} vertices")

    results = {
        "source_of_truth": {
            "gt_pos": GT_POS.tolist(),
            "voxel_size_mm": VOXEL_SIZE_MM,
            "volume_shape": list(VOLUME_SHAPE_XYZ),
            "fluence_path": str(fluence_path),
            "vertex_extraction": "skimage.measure.marching_cubes",
            "direct_path_function": "is_direct_path_vertex (ray-march, step=0.1mm)",
            "green_function": "G_inf (semi-infinite Green's function)",
            "scale_formula": "scale_factor_logmse (geomean)",
        },
        "mask_counts": {
            "mask_direct": int(np.sum(mask_direct)),
            "mask_all": int(np.sum(mask_all)),
        },
    }

    for mask_name, mask in [("direct", mask_direct), ("all", mask_all)]:
        print(f"\n--- Mask {mask_name.upper()} ---")

        meas_masked = measurement[mask]
        green_masked = green[mask]
        dist_masked = distances[mask]
        angle_masked = angle_cos[mask]
        verts_masked = vertices_mm[mask]
        pc_masked = [path_classes[i] for i in range(n_vertices) if mask[i]]
        pl_masked = [path_lengths_list[i] for i in range(n_vertices) if mask[i]]

        scale = scale_factor_logmse(meas_masked, green_masked)
        print(f"Scale k: {scale:.4e}")

        residual = compute_residual(meas_masked, green_masked, scale)
        stats = residual_stats(residual)
        print(f"Residual mean: {stats['mean']:.4f}")
        print(f"Residual std: {stats['std']:.4f}")
        print(f"Residual max|R|: {stats['max_abs']:.4f}")
        print(f"Fraction |R|<0.1: {stats['frac_lt_0.1']:.4f}")

        log_mcx = np.log10(meas_masked + EPS)
        log_green = np.log10(scale * green_masked + EPS)
        ncc = float(np.corrcoef(log_mcx, log_green)[0, 1])
        print(f"Log-space NCC: {ncc:.4f}")

        results[f"mask_{mask_name}"] = {
            "n_vertices": int(np.sum(mask)),
            "scale": float(scale),
            "log_ncc": ncc,
            "residual_stats": stats,
        }

        csv_data = []
        indices = np.where(mask)[0]
        for j, idx in enumerate(indices):
            pl = pl_masked[j]
            csv_data.append(
                {
                    "vertex_idx": int(idx),
                    "x": float(verts_masked[j, 0]),
                    "y": float(verts_masked[j, 1]),
                    "z": float(verts_masked[j, 2]),
                    "measurement": float(meas_masked[j]),
                    "green": float(green_masked[j]),
                    "scale": float(scale),
                    "residual": float(residual[j]),
                    "distance": float(dist_masked[j]),
                    "angle_cos": float(angle_masked[j]),
                    "path_class": pc_masked[j],
                    "soft_len": float(
                        pl.get(SOFT_TISSUE_LABEL, 0) + pl.get(AIR_LABEL, 0)
                    ),
                    "liver_len": float(pl.get(LIVER_LABEL, 0)),
                    "bone_len": float(pl.get(BONE_LABEL, 0)),
                    "other_len": float(
                        sum(
                            v
                            for k, v in pl.items()
                            if k
                            not in {
                                AIR_LABEL,
                                SOFT_TISSUE_LABEL,
                                LIVER_LABEL,
                                BONE_LABEL,
                            }
                        )
                    ),
                }
            )
        save_residual_csv(csv_data, output_dir / f"residual_table_{mask_name}.csv")

        plot_histogram(
            residual,
            f"Residual distribution (Mask {mask_name.upper()})",
            output_dir / f"P5_residual_hist_{mask_name}.png",
        )
        plot_residual_vs_distance(
            residual,
            dist_masked,
            f"Residual vs distance (Mask {mask_name.upper()})",
            output_dir / f"P5_residual_vs_distance_{mask_name}.png",
        )
        plot_residual_vs_angle(
            residual,
            angle_masked,
            f"Residual vs angle (Mask {mask_name.upper()})",
            output_dir / f"P5_residual_vs_angle_{mask_name}.png",
        )

        save_html_surface(
            verts_masked,
            residual,
            f"Residual surface (Mask {mask_name.upper()})",
            output_dir / f"P5_residual_surface_{mask_name}.html",
            GT_POS,
        )

        corr_dist = float(np.corrcoef(residual, dist_masked)[0, 1])
        corr_angle = float(np.corrcoef(residual, angle_masked)[0, 1])
        print(f"Correlation with distance: {corr_dist:.4f}")
        print(f"Correlation with angle: {corr_angle:.4f}")

        results[f"mask_{mask_name}"]["corr_distance"] = corr_dist
        results[f"mask_{mask_name}"]["corr_angle"] = corr_angle

        n_bins = 10
        dist_bins = np.percentile(dist_masked, np.linspace(0, 100, n_bins + 1))
        dist_bins[-1] += 0.01
        bin_means = []
        bin_stds = []
        for b in range(n_bins):
            in_bin = (dist_masked >= dist_bins[b]) & (dist_masked < dist_bins[b + 1])
            if np.sum(in_bin) > 0:
                bin_means.append(float(np.mean(residual[in_bin])))
                bin_stds.append(float(np.std(residual[in_bin])))
            else:
                bin_means.append(0.0)
                bin_stds.append(0.0)
        results[f"mask_{mask_name}"]["distance_bins"] = {
            "edges": dist_bins.tolist(),
            "means": bin_means,
            "stds": bin_stds,
        }

        residuals_by_class = {}
        for pc in set(pc_masked):
            residuals_by_class[pc] = residual[
                [i for i, p in enumerate(pc_masked) if p == pc]
            ]
        results[f"mask_{mask_name}"]["path_class_stats"] = {}
        for pc, res in residuals_by_class.items():
            results[f"mask_{mask_name}"]["path_class_stats"][pc] = {
                "n": len(res),
                "mean": float(np.mean(res)),
                "median": float(np.median(res)),
                "std": float(np.std(res)),
                "p95": float(np.percentile(res, 95)),
            }

        if mask_name == "all":
            plot_residual_by_pathclass(
                residuals_by_class, output_dir / "P5_residual_by_pathclass.png"
            )

    print("\n--- C-5: Simple interpretable correction baselines ---")
    mask = mask_all
    meas_masked = measurement[mask]
    green_masked = green[mask]
    dist_masked = distances[mask]
    pc_masked = [path_classes[i] for i in range(n_vertices) if mask[i]]

    scale = scale_factor_logmse(meas_masked, green_masked)
    residual = compute_residual(meas_masked, green_masked, scale)
    residual_std_before = float(np.std(residual))
    print(f"Residual std before correction: {residual_std_before:.4f}")

    residual_after_global = residual - np.mean(residual)
    residual_std_global = float(np.std(residual_after_global))
    explained_global = 1 - (residual_std_global / residual_std_before) ** 2
    print(
        f"Global scale correction: std={residual_std_global:.4f}, explained={explained_global:.4f}"
    )

    n_bins = 20
    dist_bins = np.percentile(dist_masked, np.linspace(0, 100, n_bins + 1))
    dist_bins[-1] += 0.01
    distance_correction = np.zeros(len(residual))
    for b in range(n_bins):
        in_bin = (dist_masked >= dist_bins[b]) & (dist_masked < dist_bins[b + 1])
        if np.sum(in_bin) > 0:
            distance_correction[in_bin] = np.mean(residual[in_bin])
    residual_after_dist = residual - distance_correction
    residual_std_dist = float(np.std(residual_after_dist))
    explained_dist = 1 - (residual_std_dist / residual_std_before) ** 2
    print(
        f"Distance correction: std={residual_std_dist:.4f}, explained={explained_dist:.4f}"
    )

    pathclass_correction = np.zeros(len(residual))
    for pc in set(pc_masked):
        in_class = [i for i, p in enumerate(pc_masked) if p == pc]
        if len(in_class) > 0:
            pathclass_correction[in_class] = np.mean(residual[in_class])
    residual_after_pathclass = residual - pathclass_correction
    residual_std_pathclass = float(np.std(residual_after_pathclass))
    explained_pathclass = 1 - (residual_std_pathclass / residual_std_before) ** 2
    print(
        f"Path-class correction: std={residual_std_pathclass:.4f}, explained={explained_pathclass:.4f}"
    )

    results["correction_baselines"] = {
        "residual_std_before": residual_std_before,
        "global_scale": {
            "std_after": residual_std_global,
            "explained_variance": explained_global,
        },
        "distance_only": {
            "std_after": residual_std_dist,
            "explained_variance": explained_dist,
        },
        "path_class_only": {
            "std_after": residual_std_pathclass,
            "explained_variance": explained_pathclass,
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        f"Mask A (direct): n={results['mask_direct']['n_vertices']}, NCC={results['mask_direct']['log_ncc']:.4f}, std={results['mask_direct']['residual_stats']['std']:.4f}"
    )
    print(
        f"Mask B (all):    n={results['mask_all']['n_vertices']}, NCC={results['mask_all']['log_ncc']:.4f}, std={results['mask_all']['residual_stats']['std']:.4f}"
    )
    print(f"\nPath class stats (Mask B):")
    for pc, stats in results["mask_all"]["path_class_stats"].items():
        print(
            f"  {pc}: n={stats['n']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}"
        )
    print(f"\nCorrection baselines:")
    print(f"  Global scale: explained={explained_global:.4f}")
    print(f"  Distance:     explained={explained_dist:.4f}")
    print(f"  Path-class:   explained={explained_pathclass:.4f}")


if __name__ == "__main__":
    main()
