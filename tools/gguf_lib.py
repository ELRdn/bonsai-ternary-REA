"""Minimal GGUF v2/v3 reader for Bonsai reverse-engineering.

Only the parts needed by the report are implemented: header, metadata KV,
tensor info, and data offset calculation.  It deliberately does not try to be
a general-purpose GGUF library.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

GGUF_MAGIC = b"GGUF"

T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = range(6)
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(6, 13)

_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}

# ggml type ids relevant to Bonsai (llama.cpp / PrismML fork, 2026)
TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0", 142: "PQ2_0", 143: "PTQ1_0",
}

# (block bytes, elements per block)
BLOCK = {
    "F32": (4, 1), "F16": (2, 1), "BF16": (2, 1),
    "Q1_0": (18, 128), "Q2_0": (18, 64), "PQ2_0": (34, 128),
    "PTQ1_0": (28, 128), "TQ1_0": (56, 256), "TQ2_0": (66, 256),
    "Q8_0": (34, 32),
}


class _Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"wanted {n} bytes, got {len(b)}")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", "replace")

    def value(self, t: int):
        if t == T_STRING:
            return self.string()
        if t == T_ARRAY:
            et = self.u32()
            n = self.u64()
            return [self.value(et) for _ in range(n)]
        if t not in _FMT:
            raise ValueError(f"unsupported GGUF value type {t}")
        return struct.unpack(_FMT[t], self.read(struct.calcsize(_FMT[t])))[0]


@dataclass
class TensorInfo:
    name: str
    dims: list[int]      # GGML order: ne[0] is the fastest (row) dimension
    type_id: int
    offset: int          # relative to the tensor data section

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type_id, str(self.type_id))

    @property
    def params(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        t = self.type_name
        if t not in BLOCK:
            raise ValueError(f"unknown block size for {t}")
        bs, be = BLOCK[t]
        return (self.params // be) * bs


@dataclass
class GGUF:
    path: str
    version: int
    metadata: dict
    tensors: list[TensorInfo]
    data_start: int
    alignment: int
    file_size: int


def read_gguf(path: str | os.PathLike) -> GGUF:
    """Read a GGUF header.  The file is opened read-only and closed."""
    with open(path, "rb") as f:
        r = _Reader(f)
        if r.read(4) != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version = r.u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors = r.u64()
        n_kv = r.u64()
        md = {}
        for _ in range(n_kv):
            k = r.string()
            md[k] = r.value(r.u32())
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            tid = r.u32()
            off = r.u64()
            tensors.append(TensorInfo(name, dims, tid, off))
        alignment = int(md.get("general.alignment", 32))
        pos = f.tell()
        data_start = (pos + alignment - 1) // alignment * alignment
        file_size = os.path.getsize(path)
    return GGUF(str(path), version, md, tensors, data_start, alignment, file_size)


def row_bytes(t: TensorInfo) -> int:
    """Bytes per row (ne[0] elements) for a quantized tensor."""
    return (t.dims[0] // BLOCK[t.type_name][1]) * BLOCK[t.type_name][0]
