#!/usr/bin/env python3
import argparse
import json
import math
import os
import pickle
import sys
import time
import unicodedata
from collections import Counter

import numpy as np

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
    np.random.seed(seed)


def softmax(x, axis=-1):
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(dy, y):
    return y * (dy - np.sum(dy * y, axis=-1, keepdims=True))


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def gelu_backward(dy, x):
    t = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    tanh_t = np.tanh(t)
    sech2 = 1.0 - tanh_t**2
    dt_dx = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x**2)
    dx = 0.5 * (1.0 + tanh_t) + 0.5 * x * sech2 * dt_dx
    return dy * dx


def swish(x):
    sig = 1.0 / (1.0 + np.exp(-x))
    return x * sig


def swish_backward(dy, x):
    sig = 1.0 / (1.0 + np.exp(-x))
    return dy * (sig + x * sig * (1.0 - sig))


def dropout_forward(x, p, train):
    if not train or p <= 0.0:
        return x, None
    mask = (np.random.rand(*x.shape) >= p).astype(x.dtype) / (1.0 - p)
    return x * mask, mask


def dropout_backward(dout, mask):
    if mask is None:
        return dout
    return dout * mask


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


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv
    out = x_hat * gamma + beta
    cache = (x_hat, inv, gamma)
    return out, cache


def layernorm_backward(dout, cache):
    x_hat, inv, gamma = cache
    dbeta = np.sum(dout, axis=(0, 1))
    dgamma = np.sum(dout * x_hat, axis=(0, 1))
    dxhat = dout * gamma
    n = x_hat.shape[-1]
    dx = (
        (1.0 / n)
        * inv
        * (
            n * dxhat
            - np.sum(dxhat, axis=-1, keepdims=True)
            - x_hat * np.sum(dxhat * x_hat, axis=-1, keepdims=True)
        )
    )
    return dx, dgamma, dbeta


def rmsnorm_forward(x, gamma, beta, eps=1e-5):
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(mean_sq + eps)
    x_hat = x * inv
    out = x_hat * gamma + beta
    cache = (x, x_hat, inv, gamma)
    return out, cache


def rmsnorm_backward(dout, cache):
    x, x_hat, inv, gamma = cache
    dbeta = np.sum(dout, axis=(0, 1))
    dgamma = np.sum(dout * x_hat, axis=(0, 1))
    dxhat = dout * gamma
    n = x.shape[-1]
    inner = np.sum(dxhat * x, axis=-1, keepdims=True)
    dx = dxhat * inv - (x * (inv**3) * inner) / n
    return dx, dgamma, dbeta


ROPE_CACHE = {}


def get_rope_sin_cos(seq_len, head_dim, dtype, base=10000.0):
    key = (seq_len, head_dim, np.dtype(dtype))
    if key in ROPE_CACHE:
        return ROPE_CACHE[key]
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
    positions = np.arange(seq_len)
    freqs = np.outer(positions, inv_freq)
    sin = np.sin(freqs).astype(dtype)
    cos = np.cos(freqs).astype(dtype)
    ROPE_CACHE[key] = (sin, cos)
    return sin, cos


def apply_rope(x, sin, cos):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def apply_rope_backward(dy, sin, cos):
    dy1 = dy[..., 0::2]
    dy2 = dy[..., 1::2]
    dx = np.empty_like(dy)
    dx[..., 0::2] = dy1 * cos + dy2 * sin
    dx[..., 1::2] = -dy1 * sin + dy2 * cos
    return dx


def self_attention_forward(x, Wq, Wk, Wv, Wo, n_heads, dropout_p=0.0, train=True):
    b, t, d = x.shape
    if d % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    head_dim = d // n_heads
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")

    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    qh = q.reshape(b, t, n_heads, head_dim).transpose(0, 2, 1, 3)
    kh = k.reshape(b, t, n_heads, head_dim).transpose(0, 2, 1, 3)
    vh = v.reshape(b, t, n_heads, head_dim).transpose(0, 2, 1, 3)

    sin, cos = get_rope_sin_cos(t, head_dim, qh.dtype)
    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]
    qh_rot = apply_rope(qh, sin, cos)
    kh_rot = apply_rope(kh, sin, cos)

    scores = np.matmul(qh_rot, kh_rot.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    mask = np.triu(np.ones((t, t), dtype=x.dtype), 1) * -1e9
    scores = scores + mask
    weights = softmax(scores, axis=-1)
    weights_drop, attn_drop_mask = dropout_forward(weights, dropout_p, train)
    attn = np.matmul(weights_drop, vh)
    attn_concat = attn.transpose(0, 2, 1, 3).reshape(b, t, d)
    out = attn_concat @ Wo
    cache = (
        x,
        qh_rot,
        kh_rot,
        vh,
        weights,
        attn_concat,
        Wq,
        Wk,
        Wv,
        Wo,
        attn_drop_mask,
        n_heads,
        sin,
        cos,
    )
    return out, cache


def self_attention_backward(dout, cache):
    (
        x,
        qh_rot,
        kh_rot,
        vh,
        weights,
        attn_concat,
        Wq,
        Wk,
        Wv,
        Wo,
        attn_drop_mask,
        n_heads,
        sin,
        cos,
    ) = cache
    b, t, d = x.shape
    head_dim = d // n_heads

    dout_flat = dout.reshape(b * t, d)
    attn_flat = attn_concat.reshape(b * t, d)
    dWo = attn_flat.T @ dout_flat
    dattn_concat = dout @ Wo.T
    dattn = dattn_concat.reshape(b, t, n_heads, head_dim).transpose(0, 2, 1, 3)

    dweights = np.matmul(dattn, vh.transpose(0, 1, 3, 2))
    if attn_drop_mask is None:
        weights_drop = weights
    else:
        weights_drop = weights * attn_drop_mask
        dweights = dweights * attn_drop_mask
    dV = np.matmul(weights_drop.transpose(0, 1, 3, 2), dattn)
    dscores = softmax_backward(dweights, weights)

    scale = 1.0 / math.sqrt(head_dim)
    dQ_rot = np.matmul(dscores, kh_rot) * scale
    dK_rot = np.matmul(dscores.transpose(0, 1, 3, 2), qh_rot) * scale
    dQ = apply_rope_backward(dQ_rot, sin, cos)
    dK = apply_rope_backward(dK_rot, sin, cos)

    dQ_concat = dQ.transpose(0, 2, 1, 3).reshape(b, t, d)
    dK_concat = dK.transpose(0, 2, 1, 3).reshape(b, t, d)
    dV_concat = dV.transpose(0, 2, 1, 3).reshape(b, t, d)

    x_flat = x.reshape(b * t, d)
    dQ_flat = dQ_concat.reshape(b * t, d)
    dK_flat = dK_concat.reshape(b * t, d)
    dV_flat = dV_concat.reshape(b * t, d)
    dWq = x_flat.T @ dQ_flat
    dWk = x_flat.T @ dK_flat
    dWv = x_flat.T @ dV_flat
    dx = dQ_concat @ Wq.T + dK_concat @ Wk.T + dV_concat @ Wv.T
    return dx, dWq, dWk, dWv, dWo


def mlp_forward(x, W1, b1, W3, b3, W2, b2, dropout_p=0.0, train=True):
    a = x @ W1 + b1
    g = x @ W3 + b3
    s = swish(g)
    h = a * s
    out = h @ W2 + b2
    out, drop_mask = dropout_forward(out, dropout_p, train)
    cache = (x, a, g, s, h, drop_mask)
    return out, cache


def mlp_backward(dout, cache, W1, W3, W2):
    x, a, g, s, h, drop_mask = cache
    b, t, _ = x.shape

    dout = dropout_backward(dout, drop_mask)
    dout_flat = dout.reshape(b * t, -1)
    h_flat = h.reshape(b * t, -1)
    dW2 = h_flat.T @ dout_flat
    db2 = dout_flat.sum(axis=0)

    dh = dout @ W2.T
    da = dh * s
    ds = dh * a
    dg = swish_backward(ds, g)
    da_flat = da.reshape(b * t, -1)
    dg_flat = dg.reshape(b * t, -1)
    x_flat = x.reshape(b * t, -1)
    dW1 = x_flat.T @ da_flat
    db1 = da_flat.sum(axis=0)
    dW3 = x_flat.T @ dg_flat
    db3 = dg_flat.sum(axis=0)

    dx = da @ W1.T + dg @ W3.T
    return dx, dW1, db1, dW3, db3, dW2, db2


def forward(idx, params, n_heads, n_layers, train=True, dropout_p=0.0):
    b, t = idx.shape
    tok = params["token_emb"][idx]
    pos = params["pos_emb"][:t]
    x = tok + pos[None, :, :]

    layers_cache = []
    for layer in range(n_layers):
        x_ln1, ln1_cache = rmsnorm_forward(
            x, params[f"ln1_g_{layer}"], params[f"ln1_b_{layer}"]
        )
        attn_out, attn_cache = self_attention_forward(
            x_ln1,
            params[f"Wq_{layer}"],
            params[f"Wk_{layer}"],
            params[f"Wv_{layer}"],
            params[f"Wo_{layer}"],
            n_heads,
            dropout_p=dropout_p,
            train=train,
        )
        x = x + attn_out

        x_ln2, ln2_cache = rmsnorm_forward(
            x, params[f"ln2_g_{layer}"], params[f"ln2_b_{layer}"]
        )
        mlp_out, mlp_cache = mlp_forward(
            x_ln2,
            params[f"W1_{layer}"],
            params[f"b1_{layer}"],
            params[f"W3_{layer}"],
            params[f"b3_{layer}"],
            params[f"W2_{layer}"],
            params[f"b2_{layer}"],
            dropout_p=dropout_p,
            train=train,
        )
        x = x + mlp_out
        layers_cache.append(
            {"ln1": ln1_cache, "attn": attn_cache, "ln2": ln2_cache, "mlp": mlp_cache}
        )

    x_lnf, lnf_cache = rmsnorm_forward(x, params["lnf_g"], params["lnf_b"])
    logits = x_lnf @ params["token_emb"].T + params["bout"]

    cache = {
        "idx": idx,
        "x_lnf": x_lnf,
        "lnf": lnf_cache,
        "layers": layers_cache,
        "t": t,
    }
    return logits, cache


def cross_entropy_loss(logits, targets, mask=None):
    b, t, v = logits.shape
    logits_flat = logits.reshape(b * t, v)
    targets_flat = targets.reshape(b * t)

    probs = softmax(logits_flat, axis=-1)
    log_probs = -np.log(probs[np.arange(b * t), targets_flat] + 1e-9)
    if mask is None:
        loss = np.mean(log_probs)
        denom = b * t
        mask_flat = None
    else:
        mask_flat = mask.reshape(b * t).astype(np.float32)
        denom = np.sum(mask_flat) + 1e-9
        loss = np.sum(log_probs * mask_flat) / denom

    dlogits = probs
    dlogits[np.arange(b * t), targets_flat] -= 1.0
    if mask_flat is not None:
        dlogits *= mask_flat[:, None]
    dlogits /= denom
    dlogits = dlogits.reshape(b, t, v)
    return loss, dlogits


def estimate_loss(
    data, mask_data, batch_size, block_size, eval_batches, params, n_heads, n_layers
):
    if eval_batches <= 0:
        return None
    losses = []
    for _ in range(eval_batches):
        x, y, mask = get_batch(data, batch_size, block_size, mask_data)
        logits, _ = forward(x, params, n_heads, n_layers, train=False, dropout_p=0.0)
        loss, _ = cross_entropy_loss(logits, y, mask=mask)
        losses.append(loss)
    return float(np.mean(losses))


def backward(dlogits, cache, params):
    grads = {}
    idx = cache["idx"]
    x_lnf = cache["x_lnf"]
    b, t, d = x_lnf.shape
    v = dlogits.shape[-1]

    x_flat = x_lnf.reshape(b * t, d)
    dlogits_flat = dlogits.reshape(b * t, v)
    dtoken_out = (x_flat.T @ dlogits_flat).T
    grads["bout"] = dlogits_flat.sum(axis=0)
    dx = (dlogits_flat @ params["token_emb"]).reshape(b, t, d)

    dx, grads["lnf_g"], grads["lnf_b"] = rmsnorm_backward(dx, cache["lnf"])

    layers_cache = cache["layers"]
    for layer in reversed(range(len(layers_cache))):
        layer_cache = layers_cache[layer]

        dmlp_out = dx
        dmlp_in, grads[f"W1_{layer}"], grads[f"b1_{layer}"], grads[
            f"W3_{layer}"
        ], grads[f"b3_{layer}"], grads[f"W2_{layer}"], grads[f"b2_{layer}"] = mlp_backward(
            dmlp_out,
            layer_cache["mlp"],
            params[f"W1_{layer}"],
            params[f"W3_{layer}"],
            params[f"W2_{layer}"],
        )
        dx_ln2, grads[f"ln2_g_{layer}"], grads[f"ln2_b_{layer}"] = rmsnorm_backward(
            dmlp_in, layer_cache["ln2"]
        )
        dx = dx + dx_ln2

        dattn_out = dx
        dattn_in, grads[f"Wq_{layer}"], grads[f"Wk_{layer}"], grads[f"Wv_{layer}"], grads[
            f"Wo_{layer}"
        ] = self_attention_backward(dattn_out, layer_cache["attn"])
        dx_ln1, grads[f"ln1_g_{layer}"], grads[f"ln1_b_{layer}"] = rmsnorm_backward(
            dattn_in, layer_cache["ln1"]
        )
        dx = dx + dx_ln1

    grads["token_emb"] = np.zeros_like(params["token_emb"])
    np.add.at(grads["token_emb"], idx, dx)
    grads["token_emb"] += dtoken_out
    grads["pos_emb"] = np.zeros_like(params["pos_emb"])
    grads["pos_emb"][: cache["t"]] = np.sum(dx, axis=0)

    return grads


def init_params(vocab_size, block_size, d_model, mlp_hidden, n_layers):
    scale = 0.02
    params = {
        "token_emb": (scale * np.random.randn(vocab_size, d_model)).astype(np.float32),
        "pos_emb": (scale * np.random.randn(block_size, d_model)).astype(np.float32),
        "lnf_g": np.ones(d_model, dtype=np.float32),
        "lnf_b": np.zeros(d_model, dtype=np.float32),
        "bout": np.zeros(vocab_size, dtype=np.float32),
    }
    for layer in range(n_layers):
        params[f"Wq_{layer}"] = (scale * np.random.randn(d_model, d_model)).astype(
            np.float32
        )
        params[f"Wk_{layer}"] = (scale * np.random.randn(d_model, d_model)).astype(
            np.float32
        )
        params[f"Wv_{layer}"] = (scale * np.random.randn(d_model, d_model)).astype(
            np.float32
        )
        params[f"Wo_{layer}"] = (scale * np.random.randn(d_model, d_model)).astype(
            np.float32
        )
        params[f"W1_{layer}"] = (scale * np.random.randn(d_model, mlp_hidden)).astype(
            np.float32
        )
        params[f"b1_{layer}"] = np.zeros(mlp_hidden, dtype=np.float32)
        params[f"W3_{layer}"] = (scale * np.random.randn(d_model, mlp_hidden)).astype(
            np.float32
        )
        params[f"b3_{layer}"] = np.zeros(mlp_hidden, dtype=np.float32)
        params[f"W2_{layer}"] = (scale * np.random.randn(mlp_hidden, d_model)).astype(
            np.float32
        )
        params[f"b2_{layer}"] = np.zeros(d_model, dtype=np.float32)
        params[f"ln1_g_{layer}"] = np.ones(d_model, dtype=np.float32)
        params[f"ln1_b_{layer}"] = np.zeros(d_model, dtype=np.float32)
        params[f"ln2_g_{layer}"] = np.ones(d_model, dtype=np.float32)
        params[f"ln2_b_{layer}"] = np.zeros(d_model, dtype=np.float32)
    return params


def init_adam(params):
    opt_state = {}
    for k, v in params.items():
        opt_state[k] = {"m": np.zeros_like(v), "v": np.zeros_like(v)}
    return opt_state


def adam_update(params, grads, opt_state, step, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    for k in params.keys():
        m = opt_state[k]["m"]
        v = opt_state[k]["v"]
        g = grads[k]
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        opt_state[k]["m"] = m
        opt_state[k]["v"] = v


def adamw_update(
    params, grads, opt_state, step, lr, weight_decay, beta1=0.9, beta2=0.999, eps=1e-8
):
    for k in params.keys():
        m = opt_state[k]["m"]
        v = opt_state[k]["v"]
        g = grads[k]
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        if weight_decay > 0.0 and params[k].ndim >= 2:
            params[k] -= lr * weight_decay * params[k]
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        opt_state[k]["m"] = m
        opt_state[k]["v"] = v


def clip_grads(grads, max_norm):
    if max_norm <= 0.0:
        return
    total = 0.0
    for g in grads.values():
        total += np.sum(g * g)
    norm = math.sqrt(total)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for k in grads.keys():
            grads[k] *= scale


def save_checkpoint(path, params, opt_state, step):
    payload = {
        "params": params,
        "opt_state": opt_state,
        "step": step,
        "rng_state": np.random.get_state(),
        "meta": {
            "norm": "rmsnorm",
            "rope": True,
            "swi_glu": True,
            "tie_embed": True,
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if "rng_state" in payload:
        np.random.set_state(payload["rng_state"])
    return payload


def get_batch(data, batch_size, block_size, mask_data=None):
    max_start = len(data) - block_size - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[s : s + block_size] for s in starts])
    y = np.stack([data[s + 1 : s + block_size + 1] for s in starts])
    if mask_data is None:
        return x, y, None
    mask = np.stack([mask_data[s + 1 : s + block_size + 1] for s in starts])
    return x, y, mask


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


def sample_from_logits(logits, temperature=1.0, top_k=0):
    temp = max(temperature, 1e-6)
    logits = logits / temp
    if 0 < top_k < logits.shape[-1]:
        idx = np.argpartition(logits, -top_k)[-top_k:]
        masked = np.full_like(logits, -1e9)
        masked[idx] = logits[idx]
        logits = masked
    probs = softmax(logits, axis=-1)
    return np.random.choice(len(probs), p=probs)


def parse_csv_floats(value):
    if not value:
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value):
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


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

    data = np.array(data, dtype=np.int64)
    if loss_on_assistant_only:
        mask = np.array(mask, dtype=np.float32)
    else:
        mask = None
    return data, mask


def generate(
    params,
    tokenizer,
    block_size,
    start,
    length,
    n_heads,
    n_layers,
    temperature=1.0,
    top_k=0,
):
    if start:
        ctx = tokenizer.encode(start)
        if not ctx:
            ctx = [0]
    else:
        ctx = [0]
    idx = np.array(ctx, dtype=np.int64)[None, :]
    for _ in range(length):
        idx_cond = idx[:, -block_size:]
        logits, _ = forward(
            idx_cond, params, n_heads, n_layers, train=False, dropout_p=0.0
        )
        next_logits = logits[:, -1, :].reshape(-1)
        next_idx = sample_from_logits(next_logits, temperature, top_k)
        idx = np.concatenate([idx, np.array([[next_idx]], dtype=np.int64)], axis=1)
    return tokenizer.decode(idx[0])


def load_text(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if os.path.exists("data.txt"):
        with open("data.txt", "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_TEXT


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
    parser.add_argument("--start", type=str, default="Cengiz Han ")
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
    parser.add_argument("--ckpt_path", type=str, default="checkpoint.pkl")
    parser.add_argument("--ckpt_every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
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

    if chat_examples is not None:
        data, mask_data = build_chat_dataset(
            chat_examples, tokenizer, loss_on_assistant_only
        )
    else:
        data = np.array(tokenizer.encode(text), dtype=np.int64)
        mask_data = None

    if len(data) < args.block_size + 2:
        raise ValueError("data is too small for the chosen block_size")
    if args.d_model % args.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if (args.d_model // args.n_heads) % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if args.n_layers < 1:
        raise ValueError("n_layers must be >= 1")

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

    params = init_params(
        tokenizer.vocab_size, args.block_size, args.d_model, args.mlp_hidden, args.n_layers
    )
    opt_state = init_adam(params)
    start_step = 0
    if args.resume and os.path.exists(args.ckpt_path):
        ckpt = load_checkpoint(args.ckpt_path)
        meta = ckpt.get("meta", {})
        if meta.get("norm") != "rmsnorm":
            raise ValueError("checkpoint norm does not match rmsnorm or meta missing")
        params = ckpt["params"]
        opt_state = ckpt["opt_state"]
        start_step = int(ckpt.get("step", 0))
        if params["token_emb"].shape[0] != tokenizer.vocab_size:
            raise ValueError("checkpoint vocab_size does not match tokenizer")
        if params["pos_emb"].shape[0] != args.block_size:
            raise ValueError("checkpoint block_size does not match args")
        if params["token_emb"].shape[1] != args.d_model:
            raise ValueError("checkpoint d_model does not match args")
        if "Wout" in params:
            raise ValueError("checkpoint uses untied embeddings; retrain")
        for layer in range(args.n_layers):
            for key in (
                f"Wq_{layer}",
                f"Wk_{layer}",
                f"Wv_{layer}",
                f"Wo_{layer}",
                f"W1_{layer}",
                f"b1_{layer}",
                f"W3_{layer}",
                f"b3_{layer}",
                f"W2_{layer}",
                f"b2_{layer}",
                f"ln1_g_{layer}",
                f"ln1_b_{layer}",
                f"ln2_g_{layer}",
                f"ln2_b_{layer}",
            ):
                if key not in params:
                    raise ValueError("checkpoint n_layers does not match args")
    elif args.resume:
        print("resume requested but checkpoint not found", file=sys.stderr)

    sample_temps = parse_csv_floats(args.sample_temps)
    sample_top_ks = parse_csv_ints(args.sample_top_ks)
    start_text = args.start
    if do_nfkc or do_tr_lower:
        start_text = normalize_text(
            start_text, nfkc=do_nfkc, turkish_lower=do_tr_lower
        )

    t0 = time.time()
    eval_every = args.eval_every if args.eval_every > 0 else args.print_every
    for step in range(start_step + 1, args.steps + 1):
        x, y, loss_mask = get_batch(
            train_data, args.batch_size, args.block_size, train_mask
        )
        logits, cache = forward(
            x, params, args.n_heads, args.n_layers, train=True, dropout_p=args.dropout
        )
        loss, dlogits = cross_entropy_loss(logits, y, mask=loss_mask)
        grads = backward(dlogits, cache, params)
        clip_grads(grads, args.grad_clip)
        if args.weight_decay > 0.0:
            adamw_update(
                params, grads, opt_state, step, args.lr, args.weight_decay
            )
        else:
            adam_update(params, grads, opt_state, step, args.lr)

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            save_checkpoint(args.ckpt_path, params, opt_state, step)
        if step % args.print_every == 0:
            elapsed = time.time() - t0
            msg = f"step {step} loss {loss:.4f}"
            if val_data is not None and step % eval_every == 0:
                val_loss = estimate_loss(
                    val_data,
                    val_mask,
                    args.batch_size,
                    args.block_size,
                    args.eval_batches,
                    params,
                    args.n_heads,
                    args.n_layers,
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
                        params,
                        tokenizer,
                        args.block_size,
                        start_text,
                        args.sample_len,
                        args.n_heads,
                        args.n_layers,
                        temperature=temp,
                        top_k=top_k,
                    )
                    print(f"\n--- sample (temp={temp}, top_k={top_k}) ---")
                    print(sample)
                    print("--------------\n")


if __name__ == "__main__":
    main()
