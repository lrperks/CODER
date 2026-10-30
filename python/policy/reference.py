"""PyTorch reference decoder: same math, same KV-cache loop, same grammar mask as cpp/runtime."""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import vedit
from python.policy.export import dequantized
from python.policy.model import CFG, CharLM

ROOT = Path(__file__).resolve().parents[2]


def load_reference(name):
    """Reference model for results/models/<name>.bin (weights from data/<name>.pt, dequantized)."""
    m = CharLM(CFG)
    m.load_state_dict(torch.load(ROOT / "data" / f"{name}.pt"))
    return dequantized(m).eval()


def new_cache(m):
    d, n = CFG["d"], CFG["max_len"]
    return {"len": 0, "k": [torch.zeros(n, d) for _ in m.blocks], "v": [torch.zeros(n, d) for _ in m.blocks]}


@torch.no_grad()
def forward_chunk(m, cache, toks):
    """Appends toks to the cache; returns logits of the last token."""
    d, h = CFG["d"], CFG["heads"]
    dh = d // h
    T, p0 = len(toks), cache["len"]
    x = m.tok.weight[torch.tensor(toks)] + m.pos.weight[p0:p0 + T]
    for l, blk in enumerate(m.blocks):
        qkv = blk.qkv(blk.ln1(x))
        cache["k"][l][p0:p0 + T] = qkv[:, d:2 * d]
        cache["v"][l][p0:p0 + T] = qkv[:, 2 * d:]
        q = qkv[:, :d].view(T, h, dh)
        K = cache["k"][l][:p0 + T].view(p0 + T, h, dh)
        V = cache["v"][l][:p0 + T].view(p0 + T, h, dh)
        scores = torch.einsum("thd,shd->hts", q, K) / math.sqrt(dh)
        mask = torch.arange(p0 + T)[None, :] > (p0 + torch.arange(T))[:, None]
        scores = scores.masked_fill(mask[None], float("-inf"))
        att = torch.einsum("hts,shd->thd", torch.softmax(scores, dim=-1), V).reshape(T, d)
        x = x + blk.proj(att)
        x = x + blk.fc2(F.gelu(blk.fc1(blk.ln2(x)), approximate="tanh"))
    cache["len"] += T
    return m.head(m.lnf(x[-1]))


def decode(m, prompt: bytes, edit: bool, max_new: int, stop_at_eos: bool = True):
    """Greedy decode matching vedit.decode(mode, temperature=0)."""
    cache = new_cache(m)
    pending, text, tokens = list(prompt), [], 0
    while tokens < max_new:
        if edit:
            allowed = vedit.grammar_allowed("".join(text))
            if not allowed or allowed == [0]:
                break
            if len(allowed) == 1:  # forced
                c = allowed[0]
                pending.append(c)
                text.append(chr(c))
                tokens += 1
                continue
        else:
            allowed = list(range(128))
        logits = forward_chunk(m, cache, pending)
        pending = []
        mask = torch.full((128,), float("-inf"))
        mask[allowed] = 0.0
        c = int(torch.argmax(logits + mask))
        if c == 0 and stop_at_eos:
            break
        pending.append(c)
        text.append(chr(c))
        tokens += 1
    return "".join(text), tokens
