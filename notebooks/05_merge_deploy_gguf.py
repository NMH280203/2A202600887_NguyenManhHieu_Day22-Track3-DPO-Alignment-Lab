#!/usr/bin/env python3
"""NB5: Merge LoRA adapter into base FP16 and export GGUF."""

from pathlib import Path

DPO_DIR = Path(__file__).parent.parent / "adapters" / "dpo"
GGUF_OUT = Path(__file__).parent.parent / "gguf"
GGUF_OUT.mkdir(parents=True, exist_ok=True)

# NOTE: NB5 requires BIGGPU tier or local GPU with >20GB VRAM.
# This is a BONUS section for students with local GPU access.
# Colab T4 users: skip this section (NB5 is not required for core 100pts).

print("NB5: Merge + GGUF export")
print("Requires: BIGGPU tier (>20GB VRAM) or local GPU")
print("Skipping on T4 (core submission does not require NB5)")
