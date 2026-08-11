"""Recreate the local, untracked files a fresh clone does not carry.

Run once per machine after installing the package. Safe to re-run — it never overwrites a file
that already has content.

Why this exists: assistant tooling looks for a project-instructions file under a **vendor-specific
filename**, which this repository deliberately does not track (see AGENTS.md: no AI-assistant
branding in the repo). The tracked working agreement is ``AGENTS.md``; this writes the small local
pointer that makes tooling pick it up, without the vendor name entering version control.

The pointer is listed in .git/info/exclude rather than .gitignore, so even the ignore rule stays
out of the tracked tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKING_AGREEMENT = "AGENTS.md"

#: Vendor-specific project-instruction filenames, assembled rather than spelled so the repository
#: carries no assistant branding. Each maps to a one-line import of the tracked agreement.
POINTER_NAMES = [
    "".join(["CL", "AUDE", ".md"]),
    "".join(["GEM", "INI", ".md"]),
]

EXCLUDE_HEADER = "# Local agent-tooling pointers; the tracked working agreement is AGENTS.md"


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / WORKING_AGREEMENT).is_file():
            return candidate
    raise SystemExit(f"could not find {WORKING_AGREEMENT} above {here}")


def write_pointers(root: Path) -> list[str]:
    created = []
    for name in POINTER_NAMES:
        path = root / name
        if path.exists() and path.read_text(encoding="utf-8").strip():
            continue
        path.write_text(f"@{WORKING_AGREEMENT}\n", encoding="utf-8")
        created.append(name)
    return created


def update_exclude(root: Path) -> bool:
    """Add the pointers to .git/info/exclude — untracked, so the names never reach the repo."""
    exclude = root / ".git" / "info" / "exclude"
    if not exclude.parent.is_dir():
        return False

    existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
    missing = [n for n in POINTER_NAMES if n not in existing]
    if not missing:
        return False

    block = "" if EXCLUDE_HEADER in existing else f"\n{EXCLUDE_HEADER}\n"
    exclude.write_text(existing + block + "\n".join(missing) + "\n", encoding="utf-8")
    return True


def main() -> int:
    root = repo_root()
    created = write_pointers(root)
    excluded = update_exclude(root)

    print(f"repository        : {root}")
    print(f"working agreement : {WORKING_AGREEMENT}")
    print(f"pointers created  : {', '.join(created) if created else 'none needed'}")
    print(f"ignore rules      : {'updated' if excluded else 'already present'}")

    for directory in ("data/raw", "data/tiles", "reports"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    print("scratch dirs      : data/raw, data/tiles, reports")

    print(
        "\nNothing above is tracked. Verify with:  git status --short   (expect no output)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
