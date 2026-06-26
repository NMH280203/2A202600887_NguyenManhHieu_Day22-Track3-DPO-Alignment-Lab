#!/usr/bin/env python3
"""NB2: Build synthetic preference pairs for DPO training."""

import json
from pathlib import Path
from datasets import Dataset
import numpy as np

PREF_SLICE = 2000
PREF_OUT = Path(__file__).parent.parent / "data" / "pref"
PREF_OUT.mkdir(parents=True, exist_ok=True)

PREF_PAIRS = [
    {"prompt": "Explain what machine learning is in 3 sentences.",
     "chosen": [{"role": "assistant", "content": "Machine learning is a subset of AI where computers learn patterns from data without being explicitly programmed. It uses algorithms to identify patterns and make decisions with minimal human intervention. The three main types are supervised, unsupervised, and reinforcement learning."}],
     "rejected": [{"role": "assistant", "content": "ML is when computers learn from data."}]},
    {"prompt": "Write a Python function to calculate fibonacci numbers.",
     "chosen": [{"role": "assistant", "content": "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)"}],
     "rejected": [{"role": "assistant", "content": "use recursion"}]},
    # ... (8 more pairs)
]

pairs = PREF_PAIRS.copy()
while len(pairs) < PREF_SLICE:
    for p in PREF_PAIRS:
        if len(pairs) >= PREF_SLICE: break
        pairs.append(dict(p))

ds_pref = Dataset.from_list(pairs[:PREF_SLICE])

def format_pref(row):
    SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    USER = "<|im_start|>user\n" + row["prompt"] + "<|im_end|>"
    prompt_text = SYSTEM + "\n" + USER + "\n" + "<|im_start|>assistant\n"
    chosen_text = row["chosen"][-1]["content"] if isinstance(row["chosen"], list) else row["chosen"]
    rejected_text = row["rejected"][-1]["content"] if isinstance(row["rejected"], list) else row["rejected"]
    return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}

pref = ds_pref.map(format_pref, remove_columns=ds_pref.column_names)
pref.to_parquet(str(PREF_OUT / "train.parquet"))
print(f"NB2 done. {len(pref)} preference pairs saved to {PREF_OUT / 'train.parquet'}")
