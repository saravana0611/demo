# Company LLM Assistant (Starter)

A minimal, production-minded assistant that wraps an open-source LLM for chat, with training scaffolding for SFT/LoRA and a tiny from-scratch GPT. Designed to be extended for your company use cases.

## Features
- FastAPI chat server with streaming
- CLI chat client
- Pluggable base model via Hugging Face
- Simple conversation memory
- Config via env or YAML
- SFT/LoRA fine-tuning script (TRL + PEFT)
- Tiny from-scratch GPT training example

## Quickstart

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Set your Hugging Face token to access gated models:
   ```bash
   export HUGGINGFACE_HUB_TOKEN=YOUR_TOKEN
   ```
4. Run the server (defaults to TinyLlama chat model; change via config):
   ```bash
   uvicorn assistant.chat_server:app --host 0.0.0.0 --port 8000
   ```
5. Chat from CLI:
   ```bash
   python -m assistant.cli
   ```

### Configuration
- Env vars or YAML file (see `configs/assistant.example.yaml`).
- Key env vars:
  - `ASSISTANT_MODEL_NAME` (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
  - `ASSISTANT_MAX_NEW_TOKENS` (default 512)
  - `ASSISTANT_TEMPERATURE` (default 0.7)
  - `ASSISTANT_TOP_P` (default 0.95)
  - `ASSISTANT_USE_4BIT` (default false)
  - `ASSISTANT_MEMORY_TURNS` (default 16)
  - `ASSISTANT_CONFIG_YAML` (path to YAML config)

### Example request
```bash
curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false
}'
```

### Notes
- CPU-only will work with small models, but larger models require a GPU.
- For 4-bit quantization, install `bitsandbytes` where supported and set `ASSISTANT_USE_4BIT=true`.

## Training

### SFT/LoRA (recommended)
Fine-tune an instruction-tuned base model on your internal data.
```bash
python training/sft_lora.py \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_path /path/to/data.jsonl \
  --output_dir /path/to/output \
  --batch_size 4 --epochs 3
```
- Dataset expects JSONL with fields: `system` (optional), `user`, `assistant`.

### From-scratch tiny GPT (educational)
Train a small GPT on tokenized text to verify pipeline end-to-end.
```bash
python training/tokenizer_train.py --input_dir /data/raw --output_dir /data/tokenizer --vocab_size 32000
python training/dataset_prep.py --input_dir /data/raw --output_file /data/train.jsonl
python training/from_scratch_gpt.py --tokenizer /data/tokenizer/spm.model --train_file /data/train.jsonl --out_dir /checkpoints/tinygpt
```

## Structure
- `assistant/`: runtime assistant (server, CLI, model, memory)
- `training/`: SFT/LoRA and tiny GPT training utilities
- `configs/`: example YAML config

## License
For internal/company use. Ensure compliance with the base model and data licenses you choose.