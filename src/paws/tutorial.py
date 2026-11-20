"""Makes the course's `part1` and `part2` packages (external/nlp1-student/) importable. Import tutorial code via this module."""

import sys

from paws.config import ROOT

TUTORIAL_DIR = ROOT / "external" / "nlp1-student"

if not (TUTORIAL_DIR / "part1").is_dir() or not (TUTORIAL_DIR / "part2").is_dir():
    raise ImportError(f"Course code not found at {TUTORIAL_DIR}; run the git clone + checkout from README.md")

sys.path.insert(0, str(TUTORIAL_DIR))

import part1
import part2
