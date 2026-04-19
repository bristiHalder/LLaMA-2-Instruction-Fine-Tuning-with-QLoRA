import modal

app = modal.App("llama-finetune")

# Environment
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
        "sentencepiece",
        "huggingface_hub"
    )
)

# Secrets + storage
secret = modal.Secret.from_name("huggingface")
volume = modal.Volume.from_name("llama-vol", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=86400,
    secrets=[secret],
    volumes={"/data": volume}
)
def train():
    import os
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments
    )
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from huggingface_hub import login

    # Login
    login(os.environ["HUGGINGFACE_TOKEN"])

    print("GPU:", torch.cuda.get_device_name(0))

    # ---------------- CONFIG ----------------
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    dataset_name = "mlabonne/guanaco-llama2-1k"
    output_dir = "/data/results"
    new_model = "/data/llama-finetuned"

    # ---------------- QUANTIZATION ----------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ---------------- DATA ----------------
    dataset = load_dataset(dataset_name, split="train")

    # ---------------- MODEL ----------------
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    # ---------------- TOKENIZER ----------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---------------- LoRA ----------------
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---------------- TRAINING ----------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=25,
        save_steps=50,
        save_total_limit=2,
        max_grad_norm=0.3,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    def formatting_func(example):
        return example["text"]

    # ---------------- TRAINER ----------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # ---------------- TRAIN ----------------
    trainer.train()

    # ---------------- SAVE ----------------
    trainer.model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)

    print("Training complete. Model saved at:", new_model)


@app.local_entrypoint()
def main():
    train.remote()