#!/usr/bin/env python3
"""Bonsai GGUF reverse-engineering helper.

Commands:
  header  <model.gguf>
  tensor  <model.gguf> --name blk.0.ffn_gate.weight --rows 64
  verify  <a.gguf> <b.gguf> --name blk.0.ffn_gate.weight --rows 64

The decoders are bit-exact with the llama.cpp reference implementations in
PrismML-Eng/llama.cpp (ggml-quants.c) as of October 2026.
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter

import numpy as np

from gguf_lib import BLOCK, TYPE_NAMES, read_gguf, row_bytes


def decode_ptq1_0(buf: bytes, nblocks: int):
    """PTQ1_0: 28-byte block, 128 ternary weights, fp16 scale at the end."""
    pow3 = (1, 3, 9, 27, 81, 243)
    t = np.zeros((nblocks, 128), np.int8)
    s = np.zeros(nblocks, np.float32)
    for b in range(nblocks):
        base = b * 28
        s[b] = struct.unpack("<e", buf[base + 26:base + 28])[0]
        qs = buf[base:base + 24]
        qh = buf[base + 24:base + 26]
        vals = []
        for n in range(5):
            for m in range(16):
                q = (int(qs[m]) * pow3[n]) & 0xFF
                vals.append(((q * 3) >> 8) - 1)
        for n in range(5):
            for m in range(8):
                q = (int(qs[16 + m]) * pow3[n]) & 0xFF
                vals.append(((q * 3) >> 8) - 1)
        for n in range(4):
            for h in range(2):
                q = (int(qh[h]) * pow3[n]) & 0xFF
                vals.append(((q * 3) >> 8) - 1)
        t[b] = vals
    return s, t


def decode_pq2_0(buf: bytes, nblocks: int):
    """PQ2_0: 34-byte block, 128 two-bit slots, fp16 scale first."""
    t = np.zeros((nblocks, 128), np.int8)
    s = np.zeros(nblocks, np.float32)
    for b in range(nblocks):
        base = b * 34
        s[b] = struct.unpack("<e", buf[base:base + 2])[0]
        qs = buf[base + 2:base + 34]
        for j in range(128):
            q = (qs[j // 4] >> ((j % 4) * 2)) & 3
            t[b, j] = q - 1
    return s, t


def decode_q2_0(buf: bytes, nblocks: int):
    """Q2_0 g64: 18-byte block, 64 two-bit slots."""
    t = np.zeros((nblocks, 64), np.int8)
    s = np.zeros(nblocks, np.float32)
    for b in range(nblocks):
        base = b * 18
        s[b] = struct.unpack("<e", buf[base:base + 2])[0]
        qs = buf[base + 2:base + 18]
        for j in range(64):
            q = (qs[j // 4] >> ((j % 4) * 2)) & 3
            t[b, j] = q - 1
    return s, t


def decode_q1_0(buf: bytes, nblocks: int):
    """Q1_0 g128: 18-byte block, 128 sign bits."""
    t = np.zeros((nblocks, 128), np.int8)
    s = np.zeros(nblocks, np.float32)
    for b in range(nblocks):
        base = b * 18
        s[b] = struct.unpack("<e", buf[base:base + 2])[0]
        qs = buf[base + 2:base + 18]
        for j in range(16):
            byte = qs[j]
            for k in range(8):
                t[b, j * 8 + k] = 1 if (byte >> k) & 1 else -1
    return s, t


DECODERS = {
    "PTQ1_0": decode_ptq1_0,
    "PQ2_0": decode_pq2_0,
    "Q2_0": decode_q2_0,
    "Q1_0": decode_q1_0,
}


def load_slice(model, name: str, rows: int):
    t = next((x for x in model.tensors if x.name == name), None)
    if t is None:
        raise SystemExit(f"tensor not found: {name}")
    if t.type_name not in DECODERS:
        raise SystemExit(f"no decoder for {t.type_name} (tensor {name})")
    rb = row_bytes(t)
    with open(model.path, "rb") as f:
        f.seek(model.data_start + t.offset)
        buf = f.read(rows * rb)
    if len(buf) != rows * rb:
        raise SystemExit("short read")
    bs, be = BLOCK[t.type_name]
    blocks_per_row = t.dims[0] // be
    s, tr = DECODERS[t.type_name](buf, rows * blocks_per_row)
    return t, s.reshape(rows, blocks_per_row), tr.reshape(rows, blocks_per_row, be)


def cmd_header(args):
    m = read_gguf(args.model)
    total_bytes = 0
    total_params = 0
    counts = Counter()
    for t in m.tensors:
        counts[t.type_name] += 1
        total_bytes += t.nbytes
        total_params += t.params
    print(f"path        {m.path}")
    print(f"version     {m.version}")
    print(f"tensors     {len(m.tensors)}")
    print(f"kv          {len(m.metadata)}")
    print(f"data_start  {m.data_start}")
    print(f"file_size   {m.file_size}")
    print(f"tensor_bytes {total_bytes}")
    print(f"data+bytes  {m.data_start + total_bytes}")
    print(f"params      {total_params}")
    print(f"bpw         {total_bytes * 8 / total_params:.4f}")
    print("types       " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("\nmetadata:")
    for k in sorted(m.metadata):
        v = m.metadata[k]
        if isinstance(v, list) and len(v) > 12:
            v = f"[{len(v)} items] {v[:6]} ..."
        print(f"  {k} = {v}")


def cmd_tensor(args):
    m = read_gguf(args.model)
    t, s, tr = load_slice(m, args.name, args.rows)
    flat = tr.reshape(-1, tr.shape[-1])
    z = (flat == 0).mean()
    print(f"tensor      {t.name}")
    print(f"type        {t.type_name}")
    print(f"dims        {t.dims} (ne[0]={t.dims[0]})")
    print(f"rows        {args.rows}")
    print(f"scales      mean={s.mean():.6g} min={s.min():.6g} max={s.max():.6g}")
    print(f"zero_rate   {z:.6f}")
    print(f"nonzero     -1={np.mean(flat == -1):.4f}  0={z:.4f}  +1={np.mean(flat == 1):.4f}")
    print(f"first trits {flat[0, :32].tolist()}")
    print(f"first scales {s[0, :6].tolist()}")


def cmd_verify(args):
    ma = read_gguf(args.a)
    mb = read_gguf(args.b)
    ta, sa, xa = load_slice(ma, args.name, args.rows)
    tb, sb, xb = load_slice(mb, args.name, args.rows)
    if xa.shape != xb.shape:
        print(f"shape mismatch: {xa.shape} vs {xb.shape}")
        return
    print(f"trit match  {np.mean(xa == xb):.6f}")
    print(f"scale max rel diff {np.max(np.abs(sa - sb) / np.maximum(np.abs(sa), 1e-12)):.6g}")
    if xa.shape[1] == xb.shape[1]:
        print(f"scale exact {np.array_equal(sa, sb)}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("header")
    h.add_argument("model")
    h.set_defaults(func=cmd_header)

    t = sub.add_parser("tensor")
    t.add_argument("model")
    t.add_argument("--name", required=True)
    t.add_argument("--rows", type=int, default=8)
    t.set_defaults(func=cmd_tensor)

    v = sub.add_parser("verify")
    v.add_argument("a")
    v.add_argument("b")
    v.add_argument("--name", required=True)
    v.add_argument("--rows", type=int, default=8)
    v.set_defaults(func=cmd_verify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
