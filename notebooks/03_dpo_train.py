#!/usr/bin/env python3
"""NB3: DPO training with beta=0.1 on preference pairs."""

import json, time, torch, matplotlib.pyplot as plt
from pathlib import Path
from unsloth import FastLanguageModel
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
MAX_LEN = 512
MAX_PROMPT_LEN = 256
DPO_BETA = 0.1
DPO_LR = 5e-7
DPO_EPOCHS = 1
DPO_OUT = Path(__file__).parent.parent / "adapters" / "dpo"
DPO_OUT.mkdir(parents=True, exist_ok=True)
PREF_PATH = Path(__file__).parent.parent / "data" / "pref" / "train.parquet"
ADAPTER_DIR = Path(__file__).parent.parent / "adapters" / "sft-mini"

if __name__ == "__main__":
    policy, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_LEN, dtype=None, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = PeftModel.from_pretrained(policy, str(ADAPTER_DIR))
    policy = FastLanguageModel.get_peft_model(
        policy, r=16, lora_alpha=32, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    pref_ds = Dataset.from_parquet(str(PREF_PATH))
    config = DPOConfig(
        output_dir=str(DPO_OUT), max_length=MAX_LEN, max_prompt_length=MAX_PROMPT_LEN,
        beta=DPO_BETA, learning_rate=DPO_LR, num_train_epochs=DPO_EPOCHS,
        warmup_ratio=0.1, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        save_strategy="no", logging_steps=10,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        report_to="none", seed=42,
    )
    trainer = DPOTrainer(model=policy, ref_model=None, args=config,
                        train_dataset=pref_ds, processing_class=tokenizer)
    trainer.train()
    trainer.model.save_pretrained(str(DPO_OUT))

    logs = trainer.state.log_history
    chosen = [l["rewards/chosen"] for l in logs if "rewards/chosen" in l]
    rejected = [l["rewards/rejected"] for l in logs if "rewards/rejected" in l]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(chosen, "g-", label="chosen"); ax1.plot(rejected, "r-", label="rejected")
    ax1.set_title("DPO Reward Curves"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot([c-r for c,r in zip(chosen, rejected)], "b-", label="gap")
    ax2.set_title("Reward Gap (chosen - rejected)"); ax2.axhline(0, color="gray", ls="--")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("03-dpo-reward-curves.png", dpi=150)

    end_gap = chosen[-1] - rejected[-1]
    metrics = {"compute_tier":"T4","beta":DPO_BETA,"lr":DPO_LR,"epochs":DPO_EPOCHS,
               "end_chosen_reward": round(chosen[-1],4),"end_rejected_reward": round(rejected[-1],4),
               "end_reward_gap": round(end_gap,4)}
    with open(DPO_OUT / "dpo_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print(f"DPO done. Reward gap: {end_gap:+.4f}")
