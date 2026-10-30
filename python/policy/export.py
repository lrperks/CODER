"""Quantize a trained CharLM to the int8 file read by cpp/runtime/model.cpp.

Format (little-endian): magic "VEDM", int32 version=1, vocab, max_len, d, layers,
heads, ff; fp32 tok_emb[vocab][d], pos_emb[max_len][d]; per layer: fp32 ln1 g,b;
QLinear qkv, proj; fp32 ln2 g,b; QLinear fc1, fc2; then fp32 lnf g,b; QLinear head.
QLinear = int32 out, int32 in, fp32 scale[out], int8 q[out][in]; w = q * scale[row].
"""
import copy
import struct

import numpy as np
import torch


def quantize(w: torch.Tensor):
    w = w.detach().cpu().float()
    scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantized(model):
    """Copy of model whose linear weights equal the int8 values the runtime uses."""
    out = copy.deepcopy(model).cpu().float()
    for mod in out.modules():
        if isinstance(mod, torch.nn.Linear):
            q, s = quantize(mod.weight)
            mod.weight.data = q.float() * s[:, None]
    return out


def _write_qlinear(f, w):
    q, s = quantize(w)
    f.write(struct.pack("<ii", q.shape[0], q.shape[1]))
    f.write(s.numpy().astype("<f4").tobytes())
    f.write(q.numpy().astype(np.int8).tobytes())


def _write_f32(f, t):
    f.write(t.detach().cpu().float().numpy().astype("<f4").tobytes())


def export(model, path):
    c = model.cfg
    with open(path, "wb") as f:
        f.write(b"VEDM")
        f.write(struct.pack("<8i", 1, c["vocab"], c["max_len"], c["d"], c["layers"], c["heads"], c["ff"], 0))
        _write_f32(f, model.tok.weight)
        _write_f32(f, model.pos.weight)
        for b in model.blocks:
            _write_f32(f, b.ln1.weight); _write_f32(f, b.ln1.bias)
            _write_qlinear(f, b.qkv.weight)
            _write_qlinear(f, b.proj.weight)
            _write_f32(f, b.ln2.weight); _write_f32(f, b.ln2.bias)
            _write_qlinear(f, b.fc1.weight)
            _write_qlinear(f, b.fc2.weight)
        _write_f32(f, model.lnf.weight); _write_f32(f, model.lnf.bias)
        _write_qlinear(f, model.head.weight)
