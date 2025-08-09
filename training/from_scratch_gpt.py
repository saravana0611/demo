from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    n_positions: int = 512
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.key = nn.Linear(cfg.d_model, cfg.d_model)
        self.query = nn.Linear(cfg.d_model, cfg.d_model)
        self.value = nn.Linear(cfg.d_model, cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.n_heads = cfg.n_heads
        self.register_buffer("mask", torch.tril(torch.ones(cfg.n_positions, cfg.n_positions)).view(1, 1, cfg.n_positions, cfg.n_positions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.n_positions, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class JsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer_name: str, max_len: int = 512) -> None:
        super().__init__()
        self.samples: List[List[int]] = []
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ids = tok(rec["text"], truncation=True, max_length=max_len, return_tensors=None)["input_ids"]
                if len(ids) >= 2:
                    self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer name or local path")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq_len", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    vocab_size = len(tok)

    cfg = GPTConfig(vocab_size=vocab_size, n_positions=args.seq_len)
    model = TinyGPT(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ds = JsonlDataset(args.train_file, args.tokenizer, max_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=lambda batch: torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True))

    # We also need targets aligned with inputs; rebuild dataloader providing both tensors
    def collate(batch):
        xs, ys = zip(*batch)
        x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
        y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
        return x, y

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            steps += 1
        avg = total_loss / max(1, steps)
        print(f"epoch {epoch+1}: loss={avg:.4f}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "tinygpt.pt"))
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()