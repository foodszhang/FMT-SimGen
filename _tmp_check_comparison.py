#!/usr/bin/env python3
"""Check de_mcx_final_comparison.py output."""
import os
from pathlib import Path

sample_dir = Path("data/small_uniform_5samples/samples/sample_0000")
output_dir = Path("output/verification")

# Look for the comparison image
for f in sorted(output_dir.glob("*final*")):
    print(f)

# Also check what scripts exist for comparison
for f in sorted(Path("scripts").glob("*comparison*")):
    print(f"Script: {f}")

# Try running de_mcx_final_comparison.py if it exists
print("\nTrying de_mcx_final_comparison.py...")
import subprocess
result = subprocess.run(["uv", "run", "python", "scripts/de_mcx_final_comparison.py", "--sample", "sample_0000", "--samples_dir", "data/small_uniform_5samples/samples"],
    capture_output=True, text=True)
print(result.stdout[-2000:] if result.stdout else "")
print(result.stderr[-500:] if result.stderr else "")