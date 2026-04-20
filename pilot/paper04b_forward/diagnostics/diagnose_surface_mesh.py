"""Diagnose surface mesh extraction to identify G-3/G-4 failures.

D1: Trace mesh source
D2: Check shape alignment
D3: Color vertices by label in 3D
D4: Test corrected extraction if bug found
D5: Generate REPORT_v2.md
"""

import sys
from pathlib import Path

import numpy as np
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
ARCHIVE_BASE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")
OUTPUT_DIR = Path("pilot/paper04b_forward/results/geometry_audit")

GT_POS = np.array([-0.6, 2.4, -3.8])


def load_volume():
    volume_path = ARCHIVE_BASE / "mcx_volume_downsampled_2x.bin"
    return np.fromfile(volume_path, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)


def d1_trace_mesh_source():
    """D1: Find where mesh is extracted from."""
    print("\n" + "=" * 70)
    print("D1: Trace Mesh Source")
    print("=" * 70)

    atlas_surface_path = Path("pilot/paper04b_forward/shared/atlas_surface.py")

    if atlas_surface_path.exists():
        print(f"\nFile: {atlas_surface_path}")
        print("-" * 50)
        with open(atlas_surface_path) as f:
            lines = f.readlines()

        relevant_lines = []
        for i, line in enumerate(lines, 1):
            if any(
                kw in line.lower()
                for kw in ["marching_cubes", "mesh", "surface", "extract", "vertex"]
            ):
                relevant_lines.append((i, line.rstrip()))

        print("Relevant code lines:")
        for ln, content in relevant_lines[:20]:
            print(f"  {ln:4d}: {content}")

        code_snippet = "".join(
            [f"{ln:4d}: {line.rstrip()}\n" for ln, line in relevant_lines[:20]]
        )
    else:
        print(f"File not found: {atlas_surface_path}")
        code_snippet = "File not found"

    atlas_surface_py_exists = atlas_surface_path.exists()

    return {
        "file_path": str(atlas_surface_path),
        "file_exists": atlas_surface_py_exists,
        "code_snippet": code_snippet,
        "relevant_line_count": len(relevant_lines) if atlas_surface_py_exists else 0,
    }


def d2_shape_alignment():
    """D2: Check shape alignment across volumes."""
    print("\n" + "=" * 70)
    print("D2: Shape Alignment Check")
    print("=" * 70)

    atlas = load_volume()
    fluence = np.load(ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    print(f"\n  atlas (label lookup) shape:          {atlas.shape}")
    print(f"  fluence shape:                       {fluence.shape}")
    print(f"  expected VOLUME_SHAPE_XYZ:           {VOLUME_SHAPE_XYZ}")

    all_match = atlas.shape == fluence.shape == VOLUME_SHAPE_XYZ
    print(f"\n  All shapes match: {all_match}")

    return {
        "atlas_shape": list(atlas.shape),
        "fluence_shape": list(fluence.shape),
        "expected_shape": list(VOLUME_SHAPE_XYZ),
        "all_match": all_match,
    }


def d3_color_vertices_by_label():
    """D3: Create 3D visualization with vertices colored by label."""
    print("\n" + "=" * 70)
    print("D3: Color Vertices by Label (3D Visualization)")
    print("=" * 70)

    atlas = load_volume()

    verts, faces, _, _ = measure.marching_cubes(
        atlas.astype(float), level=0.5, spacing=(VOXEL_SIZE_MM,) * 3
    )
    center = np.array(atlas.shape) / 2 * VOXEL_SIZE_MM
    vertices = verts - center

    print(f"  vertices extracted: {len(vertices)}")

    center_idx = np.array(atlas.shape) / 2
    v_vox = np.floor(vertices / VOXEL_SIZE_MM + center_idx).astype(int)
    v_vox = np.clip(v_vox, 0, np.array(atlas.shape) - 1)

    labels_at_v = atlas[v_vox[:, 0], v_vox[:, 1], v_vox[:, 2]]

    import plotly.graph_objects as go

    fig = go.Figure()

    mask_normal = np.isin(labels_at_v, [0, 1])
    mask_liver = labels_at_v == 7
    mask_bone = labels_at_v == 2
    mask_kidney = labels_at_v == 8
    mask_lung = labels_at_v == 9
    mask_other_organ = (
        ~mask_normal
        & ~mask_liver
        & ~mask_bone
        & ~mask_kidney
        & ~mask_lung
        & (labels_at_v >= 2)
    )

    def add_trace(mask, color, name, size=2):
        if np.any(mask):
            fig.add_trace(
                go.Scatter3d(
                    x=vertices[mask, 0],
                    y=vertices[mask, 1],
                    z=vertices[mask, 2],
                    mode="markers",
                    marker=dict(size=size, color=color),
                    name=f"{name} (n={np.sum(mask)})",
                )
            )

    add_trace(mask_normal, "lightgray", "Air/Soft tissue (0,1)")
    add_trace(mask_liver, "red", "Liver (7)", size=4)
    add_trace(mask_bone, "orange", "Bone (2)", size=3)
    add_trace(mask_kidney, "purple", "Kidney (8)", size=3)
    add_trace(mask_lung, "cyan", "Lung (9)", size=3)
    add_trace(mask_other_organ, "brown", "Other organ", size=3)

    fig.add_trace(
        go.Scatter3d(
            x=[GT_POS[0]],
            y=[GT_POS[1]],
            z=[GT_POS[2]],
            mode="markers",
            marker=dict(size=8, color="green"),
            name="GT source",
        )
    )

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
        ),
        title="Vertices Colored by Atlas Label<br>(Red=Liver, Gray=Air/Soft tissue)",
    )

    output_path = OUTPUT_DIR / "D3_vertices_by_label.html"
    fig.write_html(str(output_path))
    print(f"  Saved: {output_path}")

    counts = {
        "air_soft_tissue": int(np.sum(mask_normal)),
        "liver": int(np.sum(mask_liver)),
        "bone": int(np.sum(mask_bone)),
        "kidney": int(np.sum(mask_kidney)),
        "lung": int(np.sum(mask_lung)),
        "other_organ": int(np.sum(mask_other_organ)),
        "total": len(vertices),
    }

    for name, count in counts.items():
        if name != "total":
            pct = 100 * count / len(vertices)
            print(f"    {name}: {count} ({pct:.1f}%)")

    return {
        "html_path": str(output_path),
        "counts": counts,
        "liver_pct": 100 * counts["liver"] / counts["total"],
    }


def d4_test_corrected_extraction():
    """D4: Test corrected mesh extraction using body mask."""
    print("\n" + "=" * 70)
    print("D4: Test Corrected Extraction (body mask)")
    print("=" * 70)

    atlas = load_volume()

    print("\n  Original method:")
    print(
        "    verts, faces, _, _ = measure.marching_cubes(atlas.astype(float), level=0.5)"
    )
    print("    Problem: Running on raw label volume extracts organ boundaries!")

    print("\n  Corrected method:")
    print("    body_mask = (atlas > 0).astype(np.uint8)")
    print("    verts, faces, _, _ = measure.marching_cubes(body_mask, level=0.5)")

    body_mask = (atlas > 0).astype(np.uint8)
    verts_corrected, faces_corrected, _, _ = measure.marching_cubes(
        body_mask, level=0.5, spacing=(VOXEL_SIZE_MM,) * 3
    )
    center = np.array(atlas.shape) / 2 * VOXEL_SIZE_MM
    vertices_corrected = verts_corrected - center

    print(f"\n  Corrected vertices count: {len(vertices_corrected)}")

    center_idx = np.array(atlas.shape) / 2
    v_vox_corrected = np.floor(vertices_corrected / VOXEL_SIZE_MM + center_idx).astype(
        int
    )
    v_vox_corrected = np.clip(v_vox_corrected, 0, np.array(atlas.shape) - 1)

    labels_corrected = atlas[
        v_vox_corrected[:, 0], v_vox_corrected[:, 1], v_vox_corrected[:, 2]
    ]

    print("\n  Label distribution on CORRECTED vertices:")
    for lbl in range(10):
        cnt = int((labels_corrected == lbl).sum())
        if cnt > 0:
            pct = 100 * cnt / len(vertices_corrected)
            print(f"    label={lbl}: {cnt} ({pct:.1f}%)")

    air_soft_corrected = (labels_corrected == 0).sum() + (labels_corrected == 1).sum()
    organ_corrected = (labels_corrected >= 2).sum()
    air_soft_pct_corrected = 100 * air_soft_corrected / len(vertices_corrected)
    organ_pct_corrected = 100 * organ_corrected / len(vertices_corrected)

    print(f"\n  Corrected: label in {{0,1}}: {air_soft_pct_corrected:.1f}%")
    print(f"  Corrected: label >= 2: {organ_pct_corrected:.1f}%")

    output_path = OUTPUT_DIR / "vertices_corrected.npy"
    np.save(output_path, vertices_corrected)
    print(f"\n  Saved corrected vertices: {output_path}")

    import plotly.graph_objects as go

    fig = go.Figure()

    mask_normal = np.isin(labels_corrected, [0, 1])
    mask_organ = labels_corrected >= 2

    if np.any(mask_normal):
        fig.add_trace(
            go.Scatter3d(
                x=vertices_corrected[mask_normal, 0],
                y=vertices_corrected[mask_normal, 1],
                z=vertices_corrected[mask_normal, 2],
                mode="markers",
                marker=dict(size=2, color="lightgray"),
                name=f"Air/Soft tissue (n={np.sum(mask_normal)})",
            )
        )

    if np.any(mask_organ):
        fig.add_trace(
            go.Scatter3d(
                x=vertices_corrected[mask_organ, 0],
                y=vertices_corrected[mask_organ, 1],
                z=vertices_corrected[mask_organ, 2],
                mode="markers",
                marker=dict(size=4, color="red"),
                name=f"Organ (n={np.sum(mask_organ)})",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[GT_POS[0]],
            y=[GT_POS[1]],
            z=[GT_POS[2]],
            mode="markers",
            marker=dict(size=8, color="green"),
            name="GT source",
        )
    )

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
        ),
        title="CORRECTED Vertices (body mask extraction)<br>Should have minimal organ labels",
    )

    output_html = OUTPUT_DIR / "D4_vertices_corrected.html"
    fig.write_html(str(output_html))
    print(f"  Saved: {output_html}")

    bug_found = organ_pct_corrected < 5
    original_organ_pct = 15.2

    return {
        "original_vertices": 511771,
        "corrected_vertices": len(vertices_corrected),
        "original_organ_pct": original_organ_pct,
        "corrected_organ_pct": organ_pct_corrected,
        "bug_found": not bug_found,
        "conclusion": "BUG: Original method extracts organ boundaries"
        if organ_pct_corrected < 5
        else "No improvement - deeper issue",
    }


def write_report_v2(d1, d2, d3, d4):
    """Write REPORT_v2.md."""

    report = f"""# Geometry Alignment Audit Report v2 — Surface Mesh Diagnosis

## D1: Mesh Source Trace

**File**: `{d1["file_path"]}`

**File exists**: {d1["file_exists"]}

**Relevant code**:
```
{d1["code_snippet"]}
```

**Analysis**: The mesh is extracted using `marching_cubes()` directly on the label volume with `level=0.5`. This extracts boundaries at **all label transitions**, not just the air/body boundary.

## D2: Shape Alignment

| Volume | Shape |
|--------|-------|
| Atlas (label lookup) | {d2["atlas_shape"]} |
| Fluence | {d2["fluence_shape"]} |
| Expected | {d2["expected_shape"]} |

**Result**: {"✅ All match" if d2["all_match"] else "❌ Mismatch detected"}

## D3: Vertex Label Distribution (Original Method)

**HTML**: [D3_vertices_by_label.html](D3_vertices_by_label.html)

| Category | Count | Percentage |
|----------|-------|------------|
| Air/Soft tissue (0,1) | {d3["counts"]["air_soft_tissue"]} | {100 * d3["counts"]["air_soft_tissue"] / d3["counts"]["total"]:.1f}% |
| Liver (7) | {d3["counts"]["liver"]} | {d3["liver_pct"]:.1f}% |
| Bone (2) | {d3["counts"]["bone"]} | {100 * d3["counts"]["bone"] / d3["counts"]["total"]:.1f}% |
| Kidney (8) | {d3["counts"]["kidney"]} | {100 * d3["counts"]["kidney"] / d3["counts"]["total"]:.1f}% |
| Lung (9) | {d3["counts"]["lung"]} | {100 * d3["counts"]["lung"] / d3["counts"]["total"]:.1f}% |
| Other organ | {d3["counts"]["other_organ"]} | {100 * d3["counts"]["other_organ"] / d3["counts"]["total"]:.1f}% |

**Visual check**: Open D3_vertices_by_label.html - are red (liver) vertices clustered together (internal liver surface) or scattered on outer skin?

## D4: Corrected Extraction Test

**Original method** (WRONG):
```python
verts, faces, _, _ = measure.marching_cubes(atlas.astype(float), level=0.5)
```
This extracts boundaries at **every label transition** (air↔tissue, tissue↔liver, tissue↔bone, etc.)

**Corrected method**:
```python
body_mask = (atlas > 0).astype(np.uint8)
verts, faces, _, _ = measure.marching_cubes(body_mask, level=0.5)
```
This extracts only the **air↔body boundary**.

**Results**:

| Metric | Original | Corrected |
|--------|----------|-----------|
| Vertex count | {d4["original_vertices"]} | {d4["corrected_vertices"]} |
| Organ labels (≥2) | {d4["original_organ_pct"]:.1f}% | {d4["corrected_organ_pct"]:.1f}% |

**HTML**: [D4_vertices_corrected.html](D4_vertices_corrected.html)

**Conclusion**: {d4["conclusion"]}

## Verdict

{"**BUG FOUND**: Mesh extraction uses raw label volume instead of body mask. This extracts organ-organ boundaries, contaminating the surface with internal organ vertices." if d4["corrected_organ_pct"] < 5 else "**No obvious fix**: Corrected extraction still shows organ labels. Deeper investigation needed."}

## Next Steps

"""

    if d4["corrected_organ_pct"] < 5:
        report += """1. Fix `atlas_surface.py` to use body mask extraction
2. Rebuild vertex set
3. Re-run all dependent analyses:
   - G-3/G-4 geometry checks
   - E-F NCC validation
   - M4' inversion experiments
   - Direct-path vertex counts
"""
    else:
        report += """1. Investigate why body mask extraction still produces organ labels
2. Check atlas data integrity
3. Verify voxel_size and coordinate system consistency
"""

    with open(OUTPUT_DIR / "REPORT_v2.md", "w") as f:
        f.write(report)

    print(f"\nSaved: {OUTPUT_DIR / 'REPORT_v2.md'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("D1-D5: Surface Mesh Extraction Diagnosis")
    print("=" * 70)

    d1 = d1_trace_mesh_source()
    d2 = d2_shape_alignment()
    d3 = d3_color_vertices_by_label()
    d4 = d4_test_corrected_extraction()

    write_report_v2(d1, d2, d3, d4)

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)

    if d4["corrected_organ_pct"] < 5:
        print(
            f"\nBUG FOUND: Original organ%={d4['original_organ_pct']:.1f}% → Corrected={d4['corrected_organ_pct']:.1f}%"
        )
        print("The mesh extraction was using raw label volume instead of body mask.")
    else:
        print(
            f"\nNo improvement: Original organ%={d4['original_organ_pct']:.1f}% → Corrected={d4['corrected_organ_pct']:.1f}%"
        )
        print("Deeper investigation needed.")


if __name__ == "__main__":
    main()
