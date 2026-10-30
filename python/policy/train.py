"""Train the edit policy (target=edit) or the full-rewrite baseline (target=baseline).

Usage: python -m python.policy.train --target edit --epochs 30
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import vedit
from python.policy.export import export
from python.policy.model import CFG, CharLM

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "data" / "samples.jsonl"
MODELS = ROOT / "results" / "models"


def pairs(rows, target):
    """(prompt bytes, target bytes + EOS) for each row. Edit targets are DSL programs."""
    out = []
    for r in rows:
        annotated = target == "edit"
        prompt = vedit.render_prompt(r["test"], r["buggy"], annotated).encode()
        y = (r["gold"] if annotated else r["fixed"]).encode() + b"\x00"
        out.append((prompt, y))
    return out


def batch(items):
    T = max(len(p) + len(y) for p, y in items)
    x = torch.zeros(len(items), T, dtype=torch.long)
    y = torch.full((len(items), T), -100, dtype=torch.long)
    for i, (p, t) in enumerate(items):
        seq = list(p) + list(t)
        x[i, :len(seq)] = torch.tensor(seq)
        y[i, len(p) - 1:len(seq) - 1] = torch.tensor(list(t))
    return x, y


def train(target, epochs, seed, bs=32, lr=2e-3):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rows = [json.loads(line) for line in SAMPLES.read_text().splitlines()]
    data = pairs([r for r in rows if r["split"] == "train"], target)
    model = CharLM(CFG)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    steps = epochs * math.ceil(len(data) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gen = torch.Generator().manual_seed(seed)
    print(f"target={target} params={n_params} train={len(data)}", flush=True)
    for ep in range(epochs):
        t0, total, count = time.time(), 0.0, 0
        order = torch.randperm(len(data), generator=gen).tolist()
        model.train()
        for i in range(0, len(order), bs):
            x, y = batch([data[j] for j in order[i:i + bs]])
            loss = F.cross_entropy(model(x).view(-1, CFG["vocab"]), y.view(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
            count += 1
        print(f"epoch {ep + 1} loss {total / count:.4f} ({time.time() - t0:.0f}s)", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ROOT / "data" / f"{target}.pt")
    export(model, MODELS / f"{target}.bin")
    print(f"exported {MODELS / (target + '.bin')} params={n_params}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["edit", "baseline"], default="edit")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.target, a.epochs, a.seed)


if __name__ == "__main__":
    main()
