#!/usr/bin/env python3
"""
H1: Magic Number Audit — find all frame literals outside frame_contract.py.

Usage:
    uv run python scripts/check_frame_literals.py

Outputs:
    docs/frame_literal_audit.md (one line per hit: path:line | value | verdict)
    Exit code 0 = clean (TODO=0), 1 = has findings needing fix.
"""
import re
import sys
from collections import namedtuple
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)
AUDIT_OUTPUT = DOCS_DIR / "frame_literal_audit.md"

# Literals to audit: (regex, description)
LITERALS = [
    (r"\b19\b", "VOLUME_CENTER_WORLD X"),
    (r"\b20\b", "VOLUME_CENTER_WORLD Y"),
    (r"\b10\.4\b", "VOLUME_CENTER_WORLD Z"),
    (r"\b38\b", "TRUNK_SIZE_MM X"),
    (r"\b40\b", "TRUNK_SIZE_MM Y"),
    (r"\b20\.8\b", "TRUNK_SIZE_MM Z"),
    (r"\b190\b", "TRUNK_GRID_SHAPE X"),
    (r"\b200\b", "TRUNK_GRID_SHAPE Y"),
    (r"\b104\b", "TRUNK_GRID_SHAPE Z"),
    (r"\b34\b", "TRUNK_OFFSET_ATLAS_MM Y"),
    (r"\b30\b", "TRUNK_OFFSET legacy Y"),
    (r"\b0\.2\b", "VOXEL_SIZE_MM"),
    (r"\b0\.1\b", "ATLAS voxel size"),
]

# Patterns that indicate the hit is a legitimate use (not a frame literal)
LEGITIMATE_PATTERNS = [
    # Import lines that pull from frame_contract
    r"(?:from|import).*frame_contract",
    r"(?:from|import)\.frame_contract",
    r"#\s*\[[0-9,.\s]+\]\s*$",  # array literal in comment
    # Version strings like "0.1.0" are not frame literals
    r'__version__\s*=\s*["\'][\d.]+["\']',
    # atlas organ label indices (digimouse labels)
    r'^\s+\d+:\s*"',
    # Reshape/transpose with tuple literals (these ARE frame-related but need replacement)
    # We'll handle these specially in the audit
]

# Whitelist: (filepath_pattern, line_fragment) → legitimate non-frame use
# Add entries here when a hit is a legitimate non-frame use (port=19090, seed=42, etc.)
WHITELIST = [
    # Example: ("scripts/some_script.py", "port=19090"),
]

Hit = namedtuple("Hit", ["path", "line_no", "line", "value", "pattern_desc", "verdict"])


def is_legitimate(line: str) -> bool:
    """Return True if line contains the literal in a legitimate non-frame context."""
    for pat in LEGITIMATE_PATTERNS:
        if re.search(pat, line):
            return True
    return False


def is_whitelisted(path: Path, line_no: int, line: str, value: str) -> bool:
    for pattern, fragment in WHITELIST:
        if re.search(pattern, str(path)) and fragment in line:
            return True
    return False


def get_python_files(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*.py")):
        # Exclude paths
        if any(ex in str(p) for ex in [
            "build/", ".venv/", "site-packages/", "__pycache__/",
            ".git/", "node_modules/", ".tox/", ".eggs/",
        ]):
            continue
        files.append(p)
    return files


def audit_file(path: Path, frame_contract_path: Path, root: Path) -> list[Hit]:
    hits = []
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return hits

    for i, line in enumerate(lines, 1):
        # Skip comment-only lines
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Skip frame_contract.py itself
        if path == frame_contract_path:
            continue
        # Check each literal
        for pattern, desc in LITERALS:
            for m in re.finditer(pattern, line):
                value = m.group()
                # Skip if in a legitimate context
                if is_legitimate(line):
                    continue
                if is_whitelisted(path, i, line, value):
                    continue
                # Avoid double-match of same value on same line
                hits.append(Hit(
                    path=str(path.relative_to(root.parent)),
                    line_no=i,
                    line=line.strip()[:120],
                    value=value,
                    pattern_desc=desc,
                    verdict="TODO",
                ))
    return hits


def main():
    root = Path(__file__).parent.parent
    frame_contract_path = root / "fmt_simgen" / "frame_contract.py"

    all_hits = []
    for py_file in get_python_files(root):
        hits = audit_file(py_file, frame_contract_path, root)
        all_hits.extend(hits)

    # Deduplicate hits (same file:line:value)
    seen = set()
    unique_hits = []
    for hit in all_hits:
        key = (hit.path, hit.line_no, hit.value)
        if key not in seen:
            seen.add(key)
            unique_hits.append(hit)

    # Write markdown table
    lines_out = ["# Frame Literal Audit (H1)\n\n"]
    lines_out.append(f"**Generated**: 2026-04-21  \n")
    lines_out.append(f"**Total raw hits**: {len(all_hits)}  \n")
    lines_out.append(f"**Unique (deduped) hits**: {len(unique_hits)}  \n\n")

    lines_out.append("## Verdict Legend\n")
    lines_out.append("- `replace_with_import`: literal must be replaced with import from `frame_contract`  \n")
    lines_out.append("- `legitimate_non_frame`: not a frame literal (add to whitelist in `check_frame_literals.py`)  \n")
    lines_out.append("- `TODO`: needs review/fix before H1 pass  \n\n")

    lines_out.append("| File | Line | Value | Pattern | Verdict | Context |\n")
    lines_out.append("|------|------|-------|---------|---------|--------|\n")

    todo_count = 0
    for hit in unique_hits:
        verdict = hit.verdict
        if verdict == "TODO":
            todo_count += 1
        lines_out.append(
            f"| `{hit.path}` | {hit.line_no} | `{hit.value}` | {hit.pattern_desc} | "
            f"{verdict} | `{hit.line}` |\n"
        )

    AUDIT_OUTPUT.write_text("".join(lines_out))

    print(f"Unique hits: {len(unique_hits)}, TODO: {todo_count}")
    print(f"Audit log: {AUDIT_OUTPUT}")
    if unique_hits:
        print("\nFirst 30 hits:")
        for hit in unique_hits[:30]:
            print(f"  {hit.path}:{hit.line_no} | {hit.value} | {hit.pattern_desc}")

    sys.exit(0 if todo_count == 0 else 1)


if __name__ == "__main__":
    main()
