#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
import unicodedata
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TEXT = (
    "this is a tiny dataset for a tiny transformer.\n"
    "we train a character level language model.\n"
    "it predicts the next character from context.\n"
    "the model is small but the ideas are big.\n"
    "practice makes patterns show up.\n"
) * 50

CHAT_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "end": "<|end|>",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def normalize_text(text, nfkc=True, turkish_lower=False):
    if text is None:
        return ""
    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    text = text.replace("I\u0307", "\u0130")
    if turkish_lower:
        text = text.replace("I", "\u0131").replace("\u0130", "i")
        text = text.lower()
        text = text.replace("i\u0307", "i")
    return text


def load_text(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if os.path.exists("data.txt"):
        with open("data.txt", "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_TEXT


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def _build_word_counts(text, lower=False, max_words=0):
    if lower:
        text = text.lower()
    words = text.split()
    if max_words > 0:
        words = words[:max_words]
    return Counter(words)


def _init_bpe_vocab(word_counts):
    vocab = {}
    for word, count in word_counts.items():
        vocab[tuple(list(word) + ["</w>"])] = count
    return vocab


def _get_pair_counts(vocab):
    pairs = {}
    for word, count in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] = pairs.get(pair, 0) + count
    return pairs


def _merge_vocab(pair, vocab):
    a, b = pair
    merged = {}
    for word, count in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(a + b)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        merged[tuple(new_word)] = count
    return merged


def train_bpe(
    text,
    vocab_size=2000,
    merges=0,
    min_freq=2,
    lower=False,
    max_words=0,
    verbose=False,
    special_tokens=None,
):
    word_counts = _build_word_counts(text, lower=lower, max_words=max_words)
    vocab = _init_bpe_vocab(word_counts)
    symbols = set()
    for word in vocab.keys():
        symbols.update(word)

    if merges <= 0:
        if vocab_size <= 0:
            merges = 1000
        else:
            merges = max(0, vocab_size - len(symbols))

    merges_list = []
    for i in range(merges):
        pairs = _get_pair_counts(vocab)
        if not pairs:
            break
        best_pair, best_count = max(pairs.items(), key=lambda kv: kv[1])
        if best_count < min_freq:
            break
        vocab = _merge_vocab(best_pair, vocab)
        merged_symbol = best_pair[0] + best_pair[1]
        symbols.add(merged_symbol)
        merges_list.append(best_pair)
        if verbose and (i + 1) % 200 == 0:
            print(f"bpe merges {i+1}, best_count {best_count}")

    tokens = sorted(symbols)
    if special_tokens:
        for tok in special_tokens:
            if tok not in tokens:
                tokens.append(tok)
    if "<unk>" not in tokens:
        tokens.append("<unk>")
    return merges_list, tokens


class BPETokenizer:
    def __init__(self, merges, tokens, lower=False, special_tokens=None):
        self.merges = [tuple(pair) for pair in merges]
        self.tokens = list(tokens)
        self.lower = lower
        self.special_tokens = set(special_tokens or [])
        for tok in self.special_tokens:
            if tok not in self.tokens:
                self.tokens.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.tokens)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.vocab_size = len(self.tokens)
        self.cache = {}

    def encode_word(self, word):
        if word in self.cache:
            return self.cache[word]
        symbols = list(word) + ["</w>"]
        for a, b in self.merges:
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        self.cache[word] = symbols
        return symbols

    def encode(self, text):
        ids = []
        for word in text.split():
            if word in self.special_tokens:
                ids.append(self.stoi[word])
                continue
            if self.lower:
                word = word.lower()
            for sym in self.encode_word(word):
                ids.append(self.stoi.get(sym, self.stoi["<unk>"]))
        return ids

    def decode(self, ids):
        tokens = [self.itos[i] for i in ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ").replace("<unk>", "?")
        return text.strip()

    def save(self, vocab_path, merges_path):
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokens": self.tokens,
                    "lower": self.lower,
                    "special_tokens": sorted(self.special_tokens),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(merges_path, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

    @classmethod
    def load(cls, vocab_path, merges_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b = line.split(" ", 1)
                merges.append((a, b))
        return cls(
            merges,
            payload["tokens"],
            lower=payload.get("lower", False),
            special_tokens=payload.get("special_tokens", []),
        )


def load_chat_examples(path, system_default):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = None
            if "messages" in obj:
                messages = obj["messages"]
            elif "instruction" in obj and "output" in obj:
                user = obj["instruction"]
                if obj.get("input"):
                    user = f"{user}\n{obj['input']}"
                messages = [
                    {"role": "system", "content": system_default},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": obj["output"]},
                ]
            elif "user" in obj and "assistant" in obj:
                messages = [
                    {"role": "system", "content": obj.get("system", system_default)},
                    {"role": "user", "content": obj["user"]},
                    {"role": "assistant", "content": obj["assistant"]},
                ]
            else:
                raise ValueError("unsupported chat schema in jsonl")

            norm_msgs = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role not in ("system", "user", "assistant"):
                    continue
                norm_msgs.append({"role": role, "content": content})
            if not norm_msgs or norm_msgs[0]["role"] != "system":
                norm_msgs.insert(0, {"role": "system", "content": system_default})
            examples.append(norm_msgs)
    return examples


def normalize_chat_examples(examples, nfkc=True, turkish_lower=False):
    for messages in examples:
        for msg in messages:
            msg["content"] = normalize_text(
                msg.get("content", ""), nfkc=nfkc, turkish_lower=turkish_lower
            )
    return examples


def build_chat_dataset(examples, tokenizer, loss_on_assistant_only):
    data = []
    mask = []
    for messages in examples:
        for msg in messages:
            role_tok = CHAT_TOKENS[msg["role"]]
            prefix_ids = tokenizer.encode(role_tok)
            data.extend(prefix_ids)
            mask.extend([0] * len(prefix_ids))

            content = msg.get("content", "").strip()
            if content:
                content_ids = tokenizer.encode(content)
                data.extend(content_ids)
                mask.extend([1 if msg["role"] == "assistant" else 0] * len(content_ids))

            if msg["role"] == "assistant":
                end_ids = tokenizer.encode(CHAT_TOKENS["end"])
                data.extend(end_ids)
                mask.extend([1] * len(end_ids))
        data.append(tokenizer.stoi.get(CHAT_TOKENS["end"], 0))
        mask.append(0)

    data = torch.tensor(data, dtype=torch.long)
    if loss_on_assistant_only:
        mask = torch.tensor(mask, dtype=torch.float32)
    else:
        mask = None
    return data, mask


def get_batch(data, batch_size, block_size, mask_data=None, device="cpu"):
    max_start = data.size(0) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,), device=data.device)
    offsets = torch.arange(block_size, device=data.device)
    idx = starts[:, None] + offsets[None, :]
    x = data[idx]
    y = data[idx + 1]
    if mask_data is None:
        return x.to(device), y.to(device), None
    mask = mask_data[idx + 1]
    return x.to(device), y.to(device), mask.to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight + self.bias


class RopeCache:
    def __init__(self, base=10000.0):
        self.base = base
        self.cache = {}

    def get(self, seq_len, head_dim, device, dtype):
        key = (seq_len, head_dim, device.type, dtype)
        if key in self.cache:
            return self.cache[key]
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
        )
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        sin = freqs.sin()
        cos = freqs.cos()
        self.cache[key] = (sin, cos)
        return sin, cos


ROPE_CACHE = RopeCache()


def apply_rope(x, sin, cos):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = d_model // n_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, d = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        sin, cos = ROPE_CACHE.get(t, self.head_dim, x.device, x.dtype)
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=x.dtype), diagonal=1
        )
        scores = scores + mask[None, None, :, :] * -1e9
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        attn = torch.matmul(weights, v)
        attn = attn.transpose(1, 2).contiguous().view(b, t, d)
        return self.wo(attn)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden)
        self.w3 = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a = self.w1(x)
        g = self.w3(x)
        out = self.w2(a * F.silu(g))
        return self.drop(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_hidden, dropout):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_hidden, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, mlp_hidden, n_layers, n_heads, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(block_size, d_model))
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, mlp_hidden, dropout) for _ in range(n_layers)]
        )
        self.ln_f = RMSNorm(d_model)
        self.bout = nn.Parameter(torch.zeros(vocab_size))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError("input length exceeds block_size")
        tok = self.token_emb(idx)
        pos = self.pos_emb[:t]
        x = tok + pos[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = F.linear(x, self.token_emb.weight, self.bout)
        return logits


def sample_from_logits(logits, temperature=1.0, top_k=0):
    temp = max(temperature, 1e-6)
    logits = logits / temp
    if 0 < top_k < logits.numel():
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits = torch.where(logits < min_val, torch.tensor(-1e9, device=logits.device), logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


@torch.no_grad()
def generate(model, tokenizer, block_size, start, length, device, temperature=1.0, top_k=0):
    model.eval()
    if start:
        ctx = tokenizer.encode(start)
        if not ctx:
            ctx = [0]
    else:
        ctx = [0]
    idx = torch.tensor(ctx, dtype=torch.long, device=device)[None, :]
    for _ in range(length):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        next_logits = logits[:, -1, :].reshape(-1)
        next_idx = sample_from_logits(next_logits, temperature, top_k)
        idx = torch.cat(
            [idx, torch.tensor([[next_idx]], dtype=torch.long, device=device)], dim=1
        )
    return tokenizer.decode(idx[0].tolist())


@torch.no_grad()
def estimate_loss(
    data, mask_data, batch_size, block_size, eval_batches, model, device
):
    if data is None:
        return None
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y, mask = get_batch(data, batch_size, block_size, mask_data, device)
        logits = model(x)
        loss = compute_loss(logits, y, mask)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def compute_loss(logits, targets, mask=None):
    b, t, v = logits.shape
    loss = F.cross_entropy(
        logits.view(b * t, v), targets.view(b * t), reduction="none"
    )
    if mask is None:
        return loss.mean()
    mask = mask.view(b * t)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def parse_csv_floats(value):
    if not value:
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value):
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def resolve_device(device_arg):
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--tokenizer", type=str, choices=["char", "bpe"], default="char")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--eval_batches", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--sample_len", type=int, default=200)
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--chat_jsonl", type=str, default="")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--loss_on_all_tokens", action="store_true")
    parser.add_argument("--bpe_merges", type=int, default=1000)
    parser.add_argument("--bpe_vocab_size", type=int, default=0)
    parser.add_argument("--bpe_min_freq", type=int, default=2)
    parser.add_argument("--bpe_lower", action="store_true")
    parser.add_argument("--bpe_max_words", type=int, default=0)
    parser.add_argument("--bpe_train", action="store_true")
    parser.add_argument("--bpe_vocab_path", type=str, default="bpe_vocab.json")
    parser.add_argument("--bpe_merges_path", type=str, default="bpe_merges.txt")
    parser.add_argument("--bpe_cache_prefix", type=str, default="")
    parser.add_argument("--sample_temps", type=str, default="")
    parser.add_argument("--sample_top_ks", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="checkpoint.pt")
    parser.add_argument("--ckpt_every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--no_unicode_normalize",
        action="store_true",
        help="disable NFKC normalization on input text",
    )
    parser.add_argument(
        "--no_turkish_lower",
        action="store_true",
        help="disable Turkish-aware lowercasing",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    text = load_text(args.data)
    do_nfkc = not args.no_unicode_normalize
    do_tr_lower = not args.no_turkish_lower
    if do_nfkc or do_tr_lower:
        text = normalize_text(text, nfkc=do_nfkc, turkish_lower=do_tr_lower)

    chat_examples = None
    loss_on_assistant_only = False
    special_tokens = []
    if args.chat_jsonl:
        if args.tokenizer != "bpe":
            raise ValueError("chat_jsonl requires --tokenizer bpe")
        system_prompt = normalize_text(
            args.system_prompt, nfkc=do_nfkc, turkish_lower=do_tr_lower
        )
        chat_examples = load_chat_examples(args.chat_jsonl, system_prompt)
        if do_nfkc or do_tr_lower:
            chat_examples = normalize_chat_examples(
                chat_examples, nfkc=do_nfkc, turkish_lower=do_tr_lower
            )
        parts = []
        for messages in chat_examples:
            for msg in messages:
                parts.append(msg.get("content", ""))
        text = "\n".join(parts)
        loss_on_assistant_only = not args.loss_on_all_tokens
        special_tokens = list(CHAT_TOKENS.values())

    if args.bpe_cache_prefix:
        args.bpe_vocab_path = f"{args.bpe_cache_prefix}_vocab.json"
        args.bpe_merges_path = f"{args.bpe_cache_prefix}_merges.txt"
    if args.tokenizer == "char":
        tokenizer = CharTokenizer(text)
    else:
        if (
            not args.bpe_train
            and os.path.exists(args.bpe_vocab_path)
            and os.path.exists(args.bpe_merges_path)
        ):
            tokenizer = BPETokenizer.load(args.bpe_vocab_path, args.bpe_merges_path)
            for tok in special_tokens:
                if tok not in tokenizer.stoi:
                    raise ValueError("special tokens missing in BPE vocab; retrain")
        else:
            merges = 0 if args.bpe_vocab_size > 0 else args.bpe_merges
            merges_list, tokens = train_bpe(
                text,
                vocab_size=args.bpe_vocab_size,
                merges=merges,
                min_freq=args.bpe_min_freq,
                lower=args.bpe_lower,
                max_words=args.bpe_max_words,
                verbose=True,
                special_tokens=special_tokens,
            )
            tokenizer = BPETokenizer(
                merges_list, tokens, lower=args.bpe_lower, special_tokens=special_tokens
            )
            tokenizer.save(args.bpe_vocab_path, args.bpe_merges_path)

    if args.d_model % args.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if (args.d_model // args.n_heads) % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if args.n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    model = TinyTransformer(
        tokenizer.vocab_size,
        args.block_size,
        args.d_model,
        args.mlp_hidden,
        args.n_layers,
        args.n_heads,
        args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_step = 0
    if args.resume and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = int(ckpt.get("step", 0))
    elif args.resume:
        print("resume requested but checkpoint not found", file=sys.stderr)

    sample_temps = parse_csv_floats(args.sample_temps)
    sample_top_ks = parse_csv_ints(args.sample_top_ks)
    start_text = args.start
    if do_nfkc or do_tr_lower:
        start_text = normalize_text(
            start_text, nfkc=do_nfkc, turkish_lower=do_tr_lower
        )

    if args.generate_only:
        if not args.resume and os.path.exists(args.ckpt_path):
            ckpt = torch.load(args.ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
        elif not os.path.exists(args.ckpt_path):
            print("checkpoint not found; using random weights", file=sys.stderr)
        temps = sample_temps or [args.temperature]
        top_ks = sample_top_ks or [args.top_k]
        for temp in temps:
            for top_k in top_ks:
                sample = generate(
                    model,
                    tokenizer,
                    args.block_size,
                    start_text,
                    args.sample_len,
                    device,
                    temperature=temp,
                    top_k=top_k,
                )
                print(f"\n--- sample (temp={temp}, top_k={top_k}) ---")
                print(sample)
                print("--------------\n")
        return

    if chat_examples is not None:
        data, mask_data = build_chat_dataset(
            chat_examples, tokenizer, loss_on_assistant_only
        )
    else:
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        mask_data = None

    if len(data) < args.block_size + 2:
        raise ValueError("data is too small for the chosen block_size")

    if args.val_split > 0.0:
        split_idx = int(len(data) * (1.0 - args.val_split))
        if split_idx <= args.block_size or (len(data) - split_idx) <= args.block_size:
            print(
                "val_split too small for block_size; disabling validation",
                file=sys.stderr,
            )
            train_data = data
            val_data = None
            train_mask = mask_data
            val_mask = None
        else:
            train_data = data[:split_idx]
            val_data = data[split_idx:]
            if mask_data is not None:
                train_mask = mask_data[:split_idx]
                val_mask = mask_data[split_idx:]
            else:
                train_mask = None
                val_mask = None
    else:
        train_data = data
        val_data = None
        train_mask = mask_data
        val_mask = None

    t0 = time.time()
    eval_every = args.eval_every if args.eval_every > 0 else args.print_every
    for step in range(start_step + 1, args.steps + 1):
        x, y, loss_mask = get_batch(
            train_data, args.batch_size, args.block_size, train_mask, device
        )
        logits = model(x)
        loss = compute_loss(logits, y, loss_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
                args.ckpt_path,
            )
        if step % args.print_every == 0:
            elapsed = time.time() - t0
            msg = f"step {step} loss {loss.item():.4f}"
            if val_data is not None and step % eval_every == 0:
                val_loss = estimate_loss(
                    val_data,
                    val_mask,
                    args.batch_size,
                    args.block_size,
                    args.eval_batches,
                    model,
                    device,
                )
                if val_loss is not None:
                    msg += f" val {val_loss:.4f}"
            print(f"{msg} time {elapsed:.1f}s")
        if step % args.sample_every == 0:
            temps = sample_temps or [args.temperature]
            top_ks = sample_top_ks or [args.top_k]
            for temp in temps:
                for top_k in top_ks:
                    sample = generate(
                        model,
                        tokenizer,
                        args.block_size,
                        start_text,
                        args.sample_len,
                        device,
                        temperature=temp,
                        top_k=top_k,
                    )
                    print(f"\n--- sample (temp={temp}, top_k={top_k}) ---")
                    print(sample)
                    print("--------------\n")


if __name__ == "__main__":
    main()
