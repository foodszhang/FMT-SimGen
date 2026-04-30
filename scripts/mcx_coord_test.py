#!/usr/bin/env python3
"""Systematic MCX coordinate debugging tests."""

import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np

MCX_BIN = "/mnt/f/win-pro/bin/mcx.exe"


def run_mcx(config, source_data, tmpdir):
    """Run MCX with config and source binary, return fluence in XYZ order."""
    vol_path = Path(tmpdir) / "vol.bin"
    source_path = Path(tmpdir) / "source.bin"

    # Create volume: 20x20x20, tissue block at center
    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    vol[4:16, 4:16, 4:16] = 1
    vol.tofile(vol_path)

    # Save source binary
    source_data.tofile(source_path)

    # Set VolumeFile to relative path
    config["Domain"]["VolumeFile"] = "vol.bin"

    config_path = Path(tmpdir) / "test.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    result = subprocess.run(
        [MCX_BIN, "-f", "test.json"],
        cwd=tmpdir, capture_output=True, text=True, timeout=60
    )

    if result.returncode != 0:
        print(f"MCX failed (rc={result.returncode}): {result.stderr[:200] if result.stderr else result.stdout[:200]}")
        return None

    # Load output
    import jdata as jd
    for f in Path(tmpdir).glob("*.jnii"):
        data = jd.load(str(f))
        nifti = data["NIFTIData"][:, :, :, 0, 0]
        # MCX Dim=[Z,Y,X] -> transpose to [X,Y,Z]
        return np.transpose(nifti, (2, 1, 0))

    return None


def test1_pos_coordinate_order():
    """Test: Is Pos = [Z, Y, X] or [X, Y, Z]?

    Setup: 1x1x1 pattern at Pos=[5, 10, 10], Dim=[20, 20, 20]
    - If Pos=[Z,Y,X]: source at Z=5, Y=10, X=10 -> expect max at (X=10, Y=10, Z=5)
    - If Pos=[X,Y,Z]: source at X=5, Y=10, Z=10 -> expect max at (X=5, Y=10, Z=10)
    """
    print("\n" + "=" * 60)
    print("TEST 1: Pos coordinate order")
    print("=" * 60)

    config = {
        "Domain": {
            "Dim": [20, 20, 20],
            "OriginType": 1,
            "LengthUnit": 0.2,
            "Media": [
                {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0, "tag": 0},
                {"mua": 0.1, "mus": 100.0, "g": 0.9, "n": 1.37, "tag": 1},
            ]
        },
        "Session": {"Photons": 50000, "RNGSeed": 42, "ID": "test1"},
        "Forward": {"T0": 0.0, "T1": 5e-08, "DT": 5e-08},
        "Optode": {
            "Source": {
                "Pos": [5, 10, 10],
                "Dir": [0, 0, 1, "_NaN_"],
                "Type": "pattern3d",
                "Pattern": {"Nx": 1, "Ny": 1, "Nz": 1, "Data": "source.bin"},
                "Param1": [1, 1, 1]
            }
        }
    }

    source_data = np.array([1.0], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        fluence = run_mcx(config, source_data, tmpdir)
        if fluence is None:
            print("FAILED: No output")
            return

        max_loc = np.unravel_index(np.argmax(fluence), fluence.shape)
        print(f"Dim: [Z={config['Domain']['Dim'][0]}, Y={config['Domain']['Dim'][1]}, X={config['Domain']['Dim'][2]}]")
        print(f"Pos (ZYX): {config['Optode']['Source']['Pos']}")
        print(f"Max fluence at XYZ voxel: {max_loc}")
        print(f"Max at physical (mm): X={max_loc[0]*0.2:.2f}, Y={max_loc[1]*0.2:.2f}, Z={max_loc[2]*0.2:.2f}")
        print(f"\nIf Pos=[Z,Y,X]: expect max at X=10, Y=10, Z=5")
        print(f"If Pos=[X,Y,Z]: expect max at X=5, Y=10, Z=10")
        print(f"Actual: X={max_loc[0]}, Y={max_loc[1]}, Z={max_loc[2]}")


def test2_pattern_dim_mapping():
    """Test: How does Pattern[Nx,Ny,Nz] map to volume axes?

    Setup: 2x2x2 pattern at Pos=[5,5,5], Dim=[20,20,20]
    Pattern with single active voxel at [px,py,pz] to trace exact mapping.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Pattern dimension mapping")
    print("=" * 60)

    positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

    for px, py, pz in positions:
        config = {
            "Domain": {
                "Dim": [20, 20, 20],
                "OriginType": 1,
                "LengthUnit": 0.2,
                "Media": [
                    {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0, "tag": 0},
                    {"mua": 0.1, "mus": 100.0, "g": 0.9, "n": 1.37, "tag": 1},
                ]
            },
            "Session": {"Photons": 50000, "RNGSeed": 42, "ID": "test2"},
            "Forward": {"T0": 0.0, "T1": 5e-08, "DT": 5e-08},
            "Optode": {
                "Source": {
                    "Pos": [5, 5, 5],
                    "Dir": [0, 0, 1, "_NaN_"],
                    "Type": "pattern3d",
                    "Pattern": {"Nx": 2, "Ny": 2, "Nz": 2, "Data": "source.bin"},
                    "Param1": [2, 2, 2]
                }
            }
        }

        # Create pattern with single active voxel
        pattern = np.zeros((2, 2, 2), dtype=np.float32)
        pattern[px, py, pz] = 1.0

        # mcx_config.py transposes: pattern_zyx = pattern.transpose(2, 1, 0)
        pattern_zyx = pattern.transpose(2, 1, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            fluence = run_mcx(config, pattern_zyx, tmpdir)
            if fluence is None:
                print(f"  [{px},{py},{pz}]: FAILED")
                continue

            max_loc = np.unravel_index(np.argmax(fluence), fluence.shape)
            # Expected: if pattern[px,py,pz] maps to volume[Pos_z+pz, Pos_y+py, Pos_x+px]
            # then max should be at X=5+px, Y=5+py, Z=5+pz
            expected = (5 + px, 5 + py, 5 + pz)
            match = max_loc == expected
            print(f"  pattern[{px},{py},{pz}] -> max at {max_loc}, expected {expected}, {'OK' if match else 'FAIL'}")


def test3_binary_layout():
    """Test: How is binary data laid out?

    Create pattern with distinct values to trace mapping.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Binary data layout")
    print("=" * 60)

    config = {
        "Domain": {
            "Dim": [20, 20, 20],
            "OriginType": 1,
            "LengthUnit": 0.2,
            "Media": [
                {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0, "tag": 0},
                {"mua": 0.1, "mus": 100.0, "g": 0.9, "n": 1.37, "tag": 1},
            ]
        },
        "Session": {"Photons": 50000, "RNGSeed": 42, "ID": "test3"},
        "Forward": {"T0": 0.0, "T1": 5e-08, "DT": 5e-08},
        "Optode": {
            "Source": {
                "Pos": [5, 5, 5],
                "Dir": [0, 0, 1, "_NaN_"],
                "Type": "pattern3d",
                "Pattern": {"Nx": 2, "Ny": 2, "Nz": 2, "Data": "source.bin"},
                "Param1": [2, 2, 2]
            }
        }
    }

    # Pattern (nx=2, ny=2, nz=2) in XYZ order
    pattern = np.zeros((2, 2, 2), dtype=np.float32)
    pattern[0, 0, 0] = 10
    pattern[1, 0, 0] = 20
    pattern[0, 1, 0] = 30
    pattern[1, 1, 0] = 40
    pattern[0, 0, 1] = 50
    pattern[1, 0, 1] = 60
    pattern[0, 1, 1] = 70
    pattern[1, 1, 1] = 80

    print("Pattern (XYZ):")
    print(f"  Z=0: {pattern[:,:,0]}")
    print(f"  Z=1: {pattern[:,:,1]}")

    # Transpose to ZYX
    pattern_zyx = pattern.transpose(2, 1, 0)
    print("Pattern (ZYX, C-order):")
    print(f"  X=0: {pattern_zyx[:,:,0]}")
    print(f"  X=1: {pattern_zyx[:,:,1]}")
    print(f"  Binary flattened: {pattern_zyx.ravel('C')}")

    with tempfile.TemporaryDirectory() as tmpdir:
        fluence = run_mcx(config, pattern_zyx, tmpdir)
        if fluence is None:
            print("FAILED: No output")
            return

        # Check where distinct values appear
        print("\nExpected mapping (Pos=[5,5,5], pattern XYZ -> volume):")
        print("  pattern[0,0,0]=10 -> vol[5+0,5+0,5+0] = vol[5,5,5]")
        print("  pattern[1,0,0]=20 -> vol[5+0,5+0,5+1] = vol[5,5,6]")
        print("  pattern[0,1,0]=30 -> vol[5+1,5+0,5+0] = vol[6,5,5]")
        print("  pattern[1,1,1]=80 -> vol[5+1,5+1,5+1] = vol[6,6,6]")

        print("\nActual fluence at expected locations:")
        for val, loc in [(10,(5,5,5)), (20,(5,5,6)), (30,(6,5,5)), (80,(6,6,6))]:
            print(f"  vol{loc} = {fluence[loc]:.1f} (expect ~{val} if direct)")


def main():
    test1_pos_coordinate_order()
    test2_pattern_dim_mapping()
    test3_binary_layout()


if __name__ == "__main__":
    main()
