# OmerGPT (tiny numpy transformer)

Minimal, educational GPT-style language model written in pure NumPy.
Includes BPE tokenizer, multi-head attention, RoPE, RMSNorm, SwiGLU,
dropout, AdamW, checkpointing, and optional chat-style datasets.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install numpy
```

## Data

Small local file:

```bash
echo "merhaba dunya" > data.txt
```

Wikipedia dump fetch + clean:

```bash
.venv/bin/python fetch_wikipedia_dump.py \
  --lang tr \
  --max_pages 2000 \
  --max_chars 20000000 \
  --min_chars 400 \
  --out data_wikipedia.txt
```

Note: `data_wikipedia.txt` and checkpoints are ignored by git (see `.gitignore`).

## BPE tokenizer (train once)

```bash
.venv/bin/python tiny_transformer_numpy.py \
  --data data_wikipedia.txt \
  --tokenizer bpe \
  --bpe_train --bpe_vocab_size 4500 --bpe_min_freq 4 --bpe_lower \
  --bpe_cache_prefix wiki_v4500 \
  --steps 0
```

## Train

```bash
.venv/bin/python tiny_transformer_numpy.py \
  --data data_wikipedia.txt \
  --tokenizer bpe \
  --bpe_cache_prefix wiki_v4500 \
  --n_layers 4 \
  --d_model 128 --mlp_hidden 256 --block_size 96 --batch_size 32 --n_heads 4 \
  --steps 15000 \
  --lr 8e-4 --dropout 0.05 --weight_decay 0.002 \
  --val_split 0.1 --eval_every 500 \
  --ckpt_every 1000 --ckpt_path wiki_ckpt_rms.pkl \
  --temperature 0.7 --top_k 40 \
  --print_every 500 --sample_every 3000
```

Resume from checkpoint:

```bash
.venv/bin/python tiny_transformer_numpy.py \
  --data data_wikipedia.txt \
  --tokenizer bpe \
  --bpe_cache_prefix wiki_v4500 \
  --resume --ckpt_path wiki_ckpt_rms.pkl
```

## Chat / instruction data

`chat_data.jsonl` supports:
- `{"messages": [{"role": "...", "content": "..."}]}`
- `{"instruction": "...", "input": "...", "output": "..."}`
- `{"system": "...", "user": "...", "assistant": "..."}`

Use it with:

```bash
.venv/bin/python tiny_transformer_numpy.py \
  --chat_jsonl chat_data.jsonl \
  --tokenizer bpe \
  --bpe_cache_prefix wiki_v4500 \
  --steps 5000
```

## Notes

- This is an educational, small-scale model. Quality improves with more data,
  longer training, and larger models (at higher compute cost).
- If you change architecture flags, old checkpoints may be incompatible.
