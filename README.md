# Fine-Tuning LLaMA 2 on an Instruction Dataset

Fine-tune Meta's LLaMA 2 7B Chat model on an instruction-following dataset using QLoRA (Quantized Low-Rank Adaptation), running entirely on Modal's cloud GPU infrastructure — no local GPU required.

---

## Overview

This project demonstrates how to efficiently fine-tune a large language model (LLM) with limited compute by combining:

- **4-bit quantization** (QLoRA via `bitsandbytes`) to drastically reduce VRAM usage
- **LoRA adapters** (via `peft`) to train only a small fraction of parameters
- **Modal** for serverless GPU execution on an NVIDIA A100 (SXM4)
- **SFTTrainer** (Supervised Fine-Tuning Trainer from `trl`) for clean, high-level training orchestration

---

## Architecture

```
+-----------------------------------------------------+
|                    Modal Cloud                      |
|  +-------------------------------------------------+ |
|  |           NVIDIA A100-SXM4-40GB GPU             | |
|  |                                                 | |
|  |  +-------------------------------------------+ | |
|  |  |         LLaMA 2 7B Chat (frozen)          | | |
|  |  |         loaded in 4-bit NF4               | | |
|  |  |                                           | | |
|  |  |    + LoRA Adapters (trainable only)       | | |
|  |  |      rank=64, alpha=16, dropout=0.1       | | |
|  |  +-------------------------------------------+ | |
|  |                      |                          | |
|  |            SFTTrainer (trl)                     | |
|  |                      |                          | |
|  |          Instruction Dataset (1,000 examples)   | |
|  +-------------------------------------------------+ |
|                                                      |
|   Secrets : HuggingFace Token                        |
|   Volume  : /data  (persistent model storage)        |
+-----------------------------------------------------+
```

### How QLoRA Works

Instead of fine-tuning all 7 billion parameters, QLoRA:

1. **Quantizes** the base model weights to **4-bit NF4** (NormalFloat4) — reducing memory by ~4x with minimal quality loss
2. **Inserts LoRA adapters** — small trainable rank-decomposition matrices — into the transformer layers
3. **Only trains the adapters** (~0.1% of total parameters), keeping the frozen base model in 4-bit

This makes it possible to fine-tune a 7B model on a single A100 GPU in under an hour.

---

## Stack

| Component | Library / Service |
|---|---|
| Base Model | `meta-llama/Llama-2-7b-chat-hf` |
| Dataset | `mlabonne/guanaco-llama2-1k` (1,000 instruction examples) |
| Quantization | `bitsandbytes` — 4-bit NF4 + BFloat16 compute |
| Parameter-Efficient Fine-Tuning | `peft` — LoRA adapters |
| Training Orchestration | `trl` — SFTTrainer |
| Training Framework | `transformers` + `accelerate` |
| Cloud Execution | `modal` — serverless GPU functions |
| GPU | NVIDIA A100-SXM4-40GB |

---

## Hyperparameters

### Quantization (`BitsAndBytesConfig`)

| Parameter | Value | Description |
|---|---|---|
| `load_in_4bit` | `True` | Load base model in 4-bit |
| `bnb_4bit_quant_type` | `nf4` | NormalFloat4 quantization scheme |
| `bnb_4bit_compute_dtype` | `bfloat16` | Compute dtype (A100 native) |

### LoRA (`LoraConfig`)

| Parameter | Value | Description |
|---|---|---|
| `r` | `64` | LoRA rank — controls adapter size |
| `lora_alpha` | `16` | Scaling factor for LoRA updates |
| `lora_dropout` | `0.1` | Dropout on adapter layers |
| `bias` | `none` | No bias terms trained |
| `task_type` | `CAUSAL_LM` | Causal language modelling |

### Training (`TrainingArguments`)

| Parameter | Value | Description |
|---|---|---|
| `num_train_epochs` | `1` | Single pass over the dataset |
| `per_device_train_batch_size` | `1` | Micro-batch size |
| `gradient_accumulation_steps` | `4` | Effective batch size = 4 |
| `learning_rate` | `2e-4` | Peak learning rate |
| `lr_scheduler_type` | `cosine` | Cosine annealing schedule |
| `warmup_steps` | `50` | Linear warmup steps |
| `max_grad_norm` | `0.3` | Gradient clipping threshold |
| `bf16` | `True` | BFloat16 mixed precision (A100) |
| `logging_steps` | `25` | Log every 25 steps |
| `save_steps` | `50` | Checkpoint every 50 steps |
| `save_total_limit` | `2` | Keep only last 2 checkpoints |

---

## Project Structure

```
Llama Fine Tuning/
├── train_llama.py      # Main fine-tuning script (Modal function)
├── test_modal.py       # Modal connection / smoke test
└── README.md
```

---

## Getting Started

### Prerequisites

1. **Modal account** — [modal.com](https://modal.com)
2. **HuggingFace account** with access to [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) (requires approval)
3. **Python 3.10+**

### Setup

```bash
# Install Modal
pip install modal

# Authenticate with Modal
modal setup

# Add your HuggingFace token as a Modal secret
modal secret create huggingface HUGGINGFACE_TOKEN=hf_your_token_here
```

### Run Training

```bash
modal run train_llama.py
```

Modal will:
1. Spin up a Debian container with all dependencies pre-installed
2. Mount your script into the container
3. Allocate an A100 GPU
4. Download the model and dataset from HuggingFace
5. Run QLoRA fine-tuning
6. Save the fine-tuned adapters to `/data/llama-finetuned` (persisted in a Modal Volume)

---

## Output

The fine-tuned LoRA adapter weights are saved to the Modal Volume at:

```
/data/llama-finetuned/   # adapter weights + tokenizer
/data/results/           # training checkpoints
```

To use the fine-tuned model, load the base model and attach the adapters:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = PeftModel.from_pretrained(base_model, "/data/llama-finetuned")
tokenizer = AutoTokenizer.from_pretrained("/data/llama-finetuned")
```

---

## References

- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288) — Touvron et al., 2023
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [LoRA Paper](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Modal Documentation](https://modal.com/docs)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Guanaco Dataset](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)
