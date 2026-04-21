"""
H3 Gate Test: verify FrameContractViolation is raised before any file is written.

Run: uv run python scripts/test_h3_assertions.py
"""
import tempfile
import shutil
from pathlib import Path
import json
import numpy as np

# Test 1: assert_focus_in_trunk raises for focus outside trunk
def test_assert_focus_outside_trunk():
    from fmt_simgen.config.asserts import FrameContractViolation, assert_focus_in_trunk

    # Valid focus (inside trunk): Y=26mm is within [0, 40]
    try:
        assert_focus_in_trunk([20.0, 26.0, 10.0], tol_mm=1.0)
        print("✅ Test 1a: valid focus accepted")
    except FrameContractViolation:
        print("❌ Test 1a: valid focus incorrectly rejected")
        return False

    # Invalid focus (outside trunk): Y=-5mm is below 0
    try:
        assert_focus_in_trunk([20.0, -5.0, 18.0], tol_mm=1.0)
        print("❌ Test 1b: invalid focus (Y=-5mm) incorrectly accepted")
        return False
    except FrameContractViolation as e:
        print(f"✅ Test 1b: invalid focus raised FrameContractViolation: {e}")

    return True


# Test 2: assert_vcw raises for wrong volume_center_world
def test_assert_vcw():
    from fmt_simgen.config.asserts import FrameContractViolation, assert_vcw

    # Wrong VCW (Y=0 instead of 20)
    try:
        assert_vcw([19.0, 0.0, 10.4], "test")
        print("❌ Test 2: wrong VCW incorrectly accepted")
        return False
    except FrameContractViolation:
        print("✅ Test 2: wrong VCW raises FrameContractViolation")

    # Correct VCW
    try:
        assert_vcw([19.0, 20.0, 10.4], "test")
        print("✅ Test 2b: correct VCW accepted")
    except FrameContractViolation:
        print("❌ Test 2b: correct VCW incorrectly rejected")
        return False

    return True


# Test 3: assert_mcx_volume_shape raises for wrong shape
def test_assert_mcx_volume_shape():
    from fmt_simgen.config.asserts import FrameContractViolation, assert_mcx_volume_shape

    # Wrong shape
    try:
        assert_mcx_volume_shape([190, 200, 100], "test")  # Z=100 instead of 104
        print("❌ Test 3a: wrong shape incorrectly accepted")
        return False
    except FrameContractViolation as e:
        print(f"✅ Test 3a: wrong shape raises: {e}")

    # Correct shape
    try:
        assert_mcx_volume_shape([190, 200, 104], "test")
        print("✅ Test 3b: correct shape accepted")
    except FrameContractViolation:
        print("❌ Test 3b: correct shape incorrectly rejected")
        return False

    return True


# Test 4: TurntableCamera raises on wrong VCW
def test_turntable_camera_vcw():
    from fmt_simgen.view_config import TurntableCamera
    from fmt_simgen.config.asserts import FrameContractViolation
    from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD

    camera = TurntableCamera()
    fake_volume = np.zeros((190, 200, 104), dtype=np.float32)

    # Wrong VCW should raise
    try:
        camera.project_volume(fake_volume, volume_center_world=[19.0, 0.0, 10.4])
        print("❌ Test 4: TurntableCamera accepted wrong VCW")
        return False
    except FrameContractViolation:
        print("✅ Test 4: TurntableCamera raises on wrong VCW")

    return True


if __name__ == "__main__":
    results = []
    results.append(("assert_focus_outside_trunk", test_assert_focus_outside_trunk()))
    results.append(("assert_vcw", test_assert_vcw()))
    results.append(("assert_mcx_volume_shape", test_assert_mcx_volume_shape()))
    results.append(("turntable_camera_vcw", test_turntable_camera_vcw()))

    print("\n=== H3 Gate Test Results ===")
    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n🎉 All H3 assertions verified. Gate H3: PASS")
    else:
        print("\n⚠️  Some H3 tests failed. Gate H3: FAIL")
        exit(1)