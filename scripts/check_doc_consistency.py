#!/usr/bin/env python3
"""
H5: Doc consistency check — grep all .md files for frame literals.

Report mismatches between doc literals and the contract constants.

Usage:
    uv run python scripts/check_doc_consistency.py
"""
import re
import sys
from pathlib import Path

ROOT = Path("/home/foods/pro/FMT-SimGen")
sys.path.insert(0, str(ROOT))

from fmt_simgen.frame_contract import (
    TRUNK_OFFSET_ATLAS_MM,
    TRUNK_SIZE_MM,
    VOXEL_SIZE_MM,
    TRUNK_GRID_SHAPE,
    VOLUME_CENTER_WORLD,
)

# Literals to check in docs (derived from contract)
FRAME_LITERALS = {
    34: ("TRUNK_OFFSET_ATLAS_MM[1]", "Y offset"),
    30: ("TRUNK_OFFSET legacy (was 30, now 34)", "Y offset"),
    19: ("VOLUME_CENTER_WORLD[0] or TRUNK_SIZE_MM[0]/2", "X half-size"),
    20: ("VOLUME_CENTER_WORLD[1] or TRUNK_SIZE_MM[1]/2", "Y half-size"),
    10.4: ("VOLUME_CENTER_WORLD[2] or TRUNK_SIZE_MM[2]/2", "Z half-size"),
    38: ("TRUNK_SIZE_MM[0]", "X size"),
    40: ("TRUNK_SIZE_MM[1]", "Y size"),
    20.8: ("TRUNK_SIZE_MM[2]", "Z size"),
    190: ("TRUNK_GRID_SHAPE[0]", "X voxels"),
    200: ("TRUNK_GRID_SHAPE[1]", "Y voxels"),
    104: ("TRUNK_GRID_SHAPE[2]", "Z voxels"),
    0.2: ("VOXEL_SIZE_MM", "voxel size"),
    0.1: ("ATLAS voxel size", "atlas voxel size"),
}

# Files/dirs to skip
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".claude"}


def check_md_file(md_path: Path) -> list:
    """Check a single .md file for frame literal mismatches."""
    issues = []
    content = md_path.read_text()
    lines = content.split("\n")

    for lineno, line in enumerate(lines, 1):
        # Skip code blocks
        if line.strip().startswith("```"):
            continue

        for literal, (name, desc) in FRAME_LITERALS.items():
            pat_str = rf"(?<![.\w]){re.escape(str(literal))}(?![.\w])"
            pattern = re.compile(pat_str)

            if pattern.search(line):
                issues.append({
                    "line": lineno,
                    "literal": literal,
                    "contract": name,
                    "desc": desc,
                    "text": line.strip()[:80],
                })

    return issues


def main():
    all_issues = []
    md_files = list(ROOT.rglob("*.md"))
    md_files = [f for f in md_files if not any(s in f.parts for s in SKIP_DIRS)]

    print("=== H5: Doc Consistency Check ===")
    print(f"Checking {len(md_files)} .md files...")
    print()

    for md_file in sorted(md_files):
        issues = check_md_file(md_file)
        if issues:
            rel = md_file.relative_to(ROOT)
            print(f"{rel}:")
            for issue in issues:
                print(f"  L{issue['line']}: literal={issue['literal']} "
                      f"({issue['desc']}) — {issue['contract']}")
                print(f"    {issue['text']}")
            print()
            all_issues.extend([(str(md_file.relative_to(ROOT)), i) for i in issues])

    if all_issues:
        print(f"FAIL: {len(all_issues)} doc literal mismatches found")
        return 1
    else:
        print("PASS: No doc literal mismatches found")
        return 0


if __name__ == "__main__":
    sys.exit(main())
