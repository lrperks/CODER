"""Char-level causal transformer (pre-LN, no biases) shared by the edit policy and the baseline."""
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = dict(vocab=128, max_len=640, d=128, layers=4, heads=4, ff=512)


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.fc1 = nn.Linear(d, ff, bias=False)
        self.fc2 = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class CharLM(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg["vocab"], cfg["d"])
        self.pos = nn.Embedding(cfg["max_len"], cfg["d"])
        self.blocks = nn.ModuleList(Block(cfg["d"], cfg["heads"], cfg["ff"]) for _ in range(cfg["layers"]))
        self.lnf = nn.LayerNorm(cfg["d"], eps=1e-5)
        self.head = nn.Linear(cfg["d"], cfg["vocab"], bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))
