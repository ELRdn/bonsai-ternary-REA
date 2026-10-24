# Evidence ledger

This directory records the identity and provenance of the artifacts used in the
report. Large model files and the full REA bundle are **not** committed.

## Artifact hashes

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `Ternary-Bonsai-2-2B-PTQ1_0.gguf` | 429,183,456 | `36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa` | **full file, matches HF `SHA256SUMS`** |
| `llama-prism-b10787-73bfe6c-bin-win-cpu-x64.zip` | 19,539,022 | `ab1ea7cfc23fdac244088fd6f1762d0efa24d3d79fe18d4f16c496cc00775728` | full file |
| `llama.dll` (inside the zip) | 3,157,504 | `08f3f83deec01ba28d93ab644f061e0513d65baaa155a69448b330423911993e` | full file |
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 | not computed (partial download) | header + tensor slices |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7,206,168,928 | not computed (partial download) | header + tensor slices |
| `Ternary-Bonsai-2-27B-F16.gguf` | 53,808,408,928 | not computed (partial download) | header + tensor slices |
| `Ternary-Bonsai-2-2B-PQ2_0.gguf` | 517,312,992 | not computed (partial download) | header + tensor slices |
| `Ternary-Bonsai-2-2B-Q2_0.gguf` | 546,689,504 | not computed (partial download) | header + tensor slices |
| `Bonsai-8B-Q1_0.gguf` | 1,158,654,496 | not computed (partial download) | header + tensor slices |

The 2B PTQ1_0 hash was verified against
[`SHA256SUMS`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/blob/main/SHA256SUMS).

## REA evidence

REA (Reverse Engineering Agent) was used for artifact identity and the package
inventory. GGUF itself is not a REA-supported binary format, so the GGUF parsing
was done with the report's own parser. The runtime ZIP/PE were inspected with
REA.

| Evidence | Subject | Notes |
|---|---|---|
| `ev_58c4c84d6e52c9e9bb04347436402f7b6fd4f85a144cb317239bb0e8d8121fc0` | ZIP | root SHA-256 `ab1ea7cf...75728` |
| `ev_ea1810fc05721036054e4f1ff45a825801a7bdac67ef625529fe40fd4944f6e2` | ZIP inventory | 75 nodes, 78 occurrences, 77 edges |
| `ev_64c79c7ee05606e40f2e8d8ab35da3de6371f8e7233b1fda5a9b78a42f376f80` | `ollama.exe` | environment PE sanity check |

The full REA bundle is ~192 MB (it embeds the artifact graph), so it is omitted.
It can be regenerated with:

```bash
rea export-evidence-bundle --path evidence/rea-evidence.json
```

## String evidence from `llama.dll`

The shipped Windows CPU build contains the following strings (extracted with a
plain ASCII scanner, not a disassembler):

```text
(guessed) PTQ1_0 - 1.75 bpw ternary (group 128)
(guessed) PQ2_0 - 2.13 bpw (group 128, legacy ftype)
(guessed) PQ2_0 - 2.13 bpw (group 128)
prism.hadamard.sign_mode is explicit but sign_widths is empty
prism.hadamard.weight_names is empty
failed to create Hadamard rotation context
```

This corroborates that the distributed runtime understands the private types
and the Hadamard metadata.

## Limitations

- The 27B files were not downloaded in full; tensor statistics come from
  byte-range slices whose offsets were computed from the GGUF header.
- No Ghidra/IDA provider was configured in this environment, so no native
  decompilation was performed. Source-level review and binary strings were used
  instead.
- The full REA bundle is not committed to keep the repository small.
