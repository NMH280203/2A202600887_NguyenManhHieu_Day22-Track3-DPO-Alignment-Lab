#!/usr/bin/env python3
"""NB1: SFT-mini — fine-tune Qwen2.5-3B-bnb-4bit on 1k Vietnamese samples."""

from pathlib import Path

BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
MAX_LEN = 512
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
SFT_SLICE = 1000
NUM_EPOCHS = 1
ADAPTER_OUT = Path(__file__).parent.parent / "adapters" / "sft-mini"
ADAPTER_OUT.mkdir(parents=True, exist_ok=True)

# Run from Colab: python notebooks/01_sft_mini.py
if __name__ == "__main__":
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    import torch, matplotlib.pyplot as plt

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_LEN, dtype=None, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    # Load + format Vietnamese data (30 samples, repeated to 1000)
    VIETNAMESE_DATA = [
        {"instruction": "Giải thích cách hoạt động của thuật toán quicksort.", "input": "", "output": "Quicksort là thuật toán sắp xếp chia để trị. Chọn một phần tử làm pivot, chia mảng thành hai phần: nhỏ hơn pivot và lớn hơn pivot. Đệ quy sắp xếp hai phần. Độ phức tạp trung bình O(n log n)."},
        {"instruction": "Viết hàm Python tính giai thừa.", "input": "", "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"},
        # ... (30 base samples, see colab notebook)
    ]
    samples = VIETNAMESE_DATA.copy()
    while len(samples) < SFT_SLICE:
        for d in VIETNAMESE_DATA:
            if len(samples) >= SFT_SLICE: break
            samples.append(dict(d))
    ds = Dataset.from_list(samples[:SFT_SLICE])

    SYSTEM = "<|im_start|>system\nBạn là trợ lý AI hữu ích.<|im_end|>"
    def format_fn(row):
        prompt = row.get("instruction","") + ("\n\n" + row["input"] if row.get("input") else "")
        return {"text": SYSTEM + "\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n" + (row.get("output") or "") + "<|im_end|>"}

    ds_formatted = ds.map(format_fn, remove_columns=ds.column_names)

    config = SFTConfig(
        output_dir=str(ADAPTER_OUT), dataset_text_field="text", max_seq_length=MAX_LEN,
        per_device_train_batch_size=PER_DEVICE_BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
        num_train_epochs=NUM_EPOCHS, fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(), logging_steps=10, save_strategy="no",
        report_to="none", seed=42,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=ds_formatted, processing_class=tokenizer)
    trainer.train()
    trainer.model.save_pretrained(str(ADAPTER_OUT))
    tokenizer.save_pretrained(str(ADAPTER_OUT))

    # Plot loss
    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    plt.figure(figsize=(8, 4)); plt.plot(losses, "b-"); plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("NB1 — SFT Training Loss"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("02-sft-loss.png", dpi=150)
    print(f"SFT done. Final loss: {losses[-1]:.4f}")
