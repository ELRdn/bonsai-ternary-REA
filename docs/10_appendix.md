## 付録A. 数式と導出

### A.1 三値のエントロピーと実効ビット幅

三値 `{-1,0,+1}` が等確率なら、情報量は

$$
H = -\sum_{i=1}^{3} \frac{1}{3}\log_2\frac{1}{3} = \log_2 3 \approx 1.58496\ \text{bit}
$$

グループサイズ `g`、スケールのビット幅 `b_s` とすると、実効ビット幅は

$$
\text{bpw} = \log_2 3 + \frac{b_s}{g}
$$

`g=128, b_s=16` なら `1.58496 + 0.125 = 1.70996`。

### A.2 PTQ1_0の実効ビット幅

1ブロック128個、28バイト。

$$
\text{bpw} = \frac{28 \times 8}{128} = 1.75
$$

内訳：

- `qs[24]` = 192ビット / 128 = 1.5 bpw
- `qh[2]` = 16ビット / 128 = 0.125 bpw
- `d[2]` = 16ビット / 128 = 0.125 bpw
- 合計 = 1.75 bpw

### A.3 PQ2_0 / Q2_0 / Q1_0の実効ビット幅

$$
\text{PQ2\_0} = \frac{34 \times 8}{128} = 2.125
$$

$$
\text{Q2\_0 g64} = \frac{18 \times 8}{64} = 2.25
$$

$$
\text{Q1\_0 g128} = \frac{18 \times 8}{128} = 1.125
$$

MLX 2-bitは36バイト/128：

$$
\text{MLX} = \frac{36 \times 8}{128} = 2.25
$$

### A.4 グループスケールの最小二乗解

三値 `t_i` が固定されたとき、`||w - s t||^2` を最小化する `s` は

$$
s^* = \frac{\sum_i w_i t_i}{\sum_i t_i^2}
$$

`t_i ∈ {-1,0,+1}` なので、`t_i^2 = |t_i|`。したがって

$$
s^* = \frac{\sum_{i: t_i \ne 0} w_i t_i}{\#\{i: t_i \ne 0\}}
$$

Bonsaiの保存スケールは、この `s^*` の1.02〜1.04倍でした。完全なLSではなく、**学習/微調整で動いた**可能性があります。

### A.5 Hadamardの直交性

Sylvester–Hadamard行列 `H_n` は

$$
H_n H_n^T = n I
$$

正規化 `H_n / √n` は直交行列。符号行列 `S` は対角で `S^2 = I`。したがって

$$
R = \frac{1}{\sqrt{n}} H S,\qquad R^{-1} = \frac{1}{\sqrt{n}} S H
$$

$$
R^{-1} R = \frac{1}{n} S H H S = \frac{1}{n} S (nI) S = I
$$

重み `W' = W R^{-1}`、活性 `x' = R x` なら `W' x' = W x`。

### A.6 PTQ1_0のbase-3パッキング

5個の三値 `t_0..t_4 ∈ {0,1,2}` を

$$
q = \sum_{k=0}^{4} t_k 3^{4-k}
$$

で `0..242` に写し、

$$
b = \left\lceil \frac{256q}{243} \right\rceil
$$

で1バイトに格納。デコードは

$$
t_k = \left\lfloor \frac{((b \cdot 3^k) \bmod 256) \cdot 3}{256} \right\rfloor
$$

`243` は `3^5`。`256/243` のスケーリングにより、243通りが256個のバイト値に単射で写ります。

### A.7 27Bのパラメータ会計

- 三値: 26,869,760,000 × 1.75 / 8 = 5,877,760,000 B
- BF16: 23,592,960 × 16 / 8 = 47,185,920 B
- F32: 2,645,504 × 32 / 8 = 10,582,016 B
- 合計: 5,935,527,936 B = 5.9355 GB
- ファイル: 5,946,648,928 B = 5.9466 GB（ヘッダ11.1MBを含む）

### A.8 理想bpwの計算

$$
\text{ideal bpw} = \frac{26{,}869{,}760{,}000 \times 1.70996 + 23{,}592{,}960 \times 16 + 2{,}645{,}504 \times 32}{26{,}895{,}998{,}464}
\approx 1.7255
$$

内訳：
- 三値コーデックの理想: `1.70996`
- PTQ1_0のbase-3パッキング: `1.75`（+0.040）
- 高精度テンソル（BF16/F32）: `+0.015`
- 実ファイル平均: `1.765`

ホワイトペーパーの「1.72」は、三値コーデック＋高精度テンソルの理想値。実際のPTQ1_0は、パッキングの端数で1.75になり、ファイル平均で1.765になります。

---

## 付録B. 再現手順

### B.1 必要なもの

- Python 3.10+
- NumPy
- Matplotlib（グラフ生成用）
- `huggingface_hub`（推奨。なくてもcurlで可）
- モデルファイル（合計で数十GB。部分取得でも可）

### B.2 モデルのダウンロード

```bash
# 2B PTQ1_0（429MB、全ファイル）
hf download prism-ml/Ternary-Bonsai-2-2B-gguf \
  Ternary-Bonsai-2-2B-PTQ1_0.gguf --local-dir artifacts

# 27B PTQ1_0（5.95GB、全ファイル。本解析では部分取得のみ使用）
# 全ファイルを落とす場合:
# hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
#   Ternary-Bonsai-2-27B-PTQ1_0.gguf --local-dir artifacts
# 部分取得は B.3 を推奨（本レポートの解析はこちら）

# ベースモデル（比較用、safetensors）
hf download Qwen/Qwen3.8-27B \
  model-00001-of-00018.safetensors --local-dir artifacts/qwen38
```

### B.3 部分取得（27Bのテンソルだけ）

GGUFのヘッダを読み、テンソルの絶対オフセットを計算してから、HTTP Rangeで必要部分だけ取得します。

```bash
# ヘッダ（先頭12MB）
curl -L -r 0-12582911 \
  https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -o artifacts/27b-head.bin

# blk.0.ffn_gate.weight の先頭64行（offsetはヘッダから計算）
curl -L -r 605245792-605317471 \
  https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -o artifacts/ptq_gate64.bin
```

> **注意**: オフセットはファイルごとに異なります。必ず`tools/bonsai_re.py header`で確認してください。

**GGUFとHFのテンソル名の対応**（27B）:

| GGUF | Hugging Face |
|---|---|
| `blk.0.ffn_gate.weight` | `model.language_model.layers.0.mlp.gate_proj.weight` |
| `blk.0.attn_qkv.weight` | `model.language_model.layers.0.linear_attn.in_proj_qkv.weight` |
| `blk.0.ssm_out.weight` | `model.language_model.layers.0.linear_attn.out_proj.weight` |
| `token_embd.weight` | `model.language_model.embed_tokens.weight` |
| `output.weight` | `lm_head.weight` |

### B.4 ヘッダの表示

```bash
python tools/bonsai_re.py header artifacts/2b-ptq1.gguf
```

出力例：

```text
version     3
tensors     320
kv          49
data_start  10999008
file_size   429183456
tensor_bytes 418184448
params      1881825088
bpw         1.7778
types       F32=169, PTQ1_0=151

metadata:
  general.architecture = qwen35
  prism.hadamard.block_size = 512
  prism.hadamard.sign_widths = [2048, 6144]
  ...
```

### B.5 テンソルのデコード

```bash
python tools/bonsai_re.py tensor artifacts/2b-ptq1.gguf \
  --name blk.0.ffn_gate.weight \
  --rows 64
```

### B.6 グラフの再生成

```bash
python tools/make_charts.py
```

`data/measurements.json` があれば、モデルファイルなしでグラフだけ再生成できます。

### B.7 ハッシュの確認

```bash
# 2B PTQ1_0
sha256sum artifacts/Ternary-Bonsai-2-2B-PTQ1_0.gguf
# 期待値: 36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa
```

---

## 付録C. 用語集

| 用語 | 意味 |
|---|---|
| **三値（ternary）** | `{-1, 0, +1}` の3値。 |
| **trit** | ternary digit。三値の1桁。 |
| **bpw** | bits per weight。重み1個あたりの実効ビット数。 |
| **グループスケール** | 128個などのグループごとに共有するFP16スケール。 |
| **QAT** | Quantization-Aware Training。訓練中に量子化を模擬する。 |
| **PTQ** | Post-Training Quantization。訓練後に量子化する。 |
| **GPTQ** | 2次近似でPTQする手法。 |
| **AWQ** | 活性感知の重み量子化。 |
| **Hadamard変換** | 直交変換の一種。外れ値を分散させる。 |
| **FWHT** | Fast Walsh–Hadamard Transform。O(n log n)の高速版。 |
| **SpinQuant / QuaRot** | 回転を使ってLLMを低bit化する手法。 |
| **PTQ1_0** | PrismMLのbase-3三値パック。1.75 bpw。 |
| **PQ2_0** | PrismMLの2-bitスロット三値パック。2.125 bpw。 |
| **Q2_0** | 上流llama.cppの2-bit型。Bonsaiではg64で2.25 bpw。 |
| **Q1_0** | 上流llama.cppの1-bit型。1.125 bpw。 |
| **GGUF** | llama.cpp系のモデルファイル形式。 |
| **dp4a** | 4個のint8積和命令。 |
| **KVキャッシュ** | 注意のKey/Valueを保存するメモリ。 |
| **hybrid attention** | linear attentionとfull attentionを混ぜたアーキテクチャ。 |
| **intelligence density** | ベンチマーク性能をサイズで割った指標。 |

---

## 参考文献

### 一次資料（PrismML）

1. PrismML. *Ternary Bonsai 2 27B*. Whitepaper, September 2026. https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf
2. PrismML. *Ternary Bonsai 2 — 2B / 4B / 9B*. Whitepaper, October 2026. https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-small-whitepaper.pdf
3. PrismML. *Ternary Bonsai 8B*. Whitepaper, April 2026. https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf
4. PrismML. *1-bit Bonsai 8B*. Whitepaper, March 2026. https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf
5. PrismML. *Bonsai 27B: binary and ternary representations*. Whitepaper, July 2026. https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf
6. PrismML. *Ternary-Bonsai-2-27B-gguf* model card. https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
7. PrismML. *Ternary Bonsai 2 27B* docs. https://docs.prismml.com/models/bonsai-2-27b
8. PrismML. *Formats & Runtime Support*. https://docs.prismml.com/download/formats
9. PrismML-Eng. *llama.cpp fork*. https://github.com/PrismML-Eng/llama.cpp
10. PrismML-Eng. *Bonsai-demo*. https://github.com/PrismML-Eng/Bonsai-demo
11. PrismML. *SHA256SUMS* (2B). https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/blob/main/SHA256SUMS

### 上流llama.cpp / GGUF

12. ggml-org. *llama.cpp*. https://github.com/ggml-org/llama.cpp
13. ggml-org. *GGUF specification*. https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
14. ggml-org. *PR #29600: Hadamard runtime*. https://github.com/ggml-org/llama.cpp/pull/29600
15. ggml-org. *PR #29672: PTQ1_0*. https://github.com/ggml-org/llama.cpp/pull/29672

### 三値量子化の理論

16. F. Li, B. Liu, X. Wang, B. Zhang. *Ternary Weight Networks*. arXiv:1605.04711, 2016.
17. C. Zhu, S. Han, H. Mao, W. J. Dally. *Trained Ternary Quantization*. arXiv:1612.01064, 2016.
18. N. Mellempudi et al. *Ternary Neural Networks with Fine-Grained Quantization*. arXiv:1705.01462, 2017.
19. S. Ma et al. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*. arXiv:2402.17764, 2024.
20. H. Wang et al. *BitNet: Scaling 1-bit Transformers for Large Language Models*. arXiv:2310.11453, 2023.
21. D. Liu, X. Liu. *Ternary Quantization: A Survey*. arXiv:2303.01505, 2023.

### PTQ / 回転 / 低bit

22. E. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. arXiv:2210.17323, 2022.
23. J. Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. arXiv:2306.00978, 2023.
24. S. Ashkboos et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*. arXiv:2404.00456, 2024.
25. Z. Liu et al. *SpinQuant: LLM quantization with learned rotations*. arXiv:2405.16406, 2024.
26. H. Xiao et al. *PTQTP: Post-Training Quantization to Trit-Planes for Large Language Models*. arXiv:2509.16989, 2025.
27. H. Huang et al. *Tequila: Trapping-free Ternary Quantization for Large Language Models*. arXiv:2509.23809, 2025.
28. H. Huang et al. *Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification*. arXiv:2601.07892, 2026.
29. E. J. Yoon. *ITQ3_S: High-Fidelity 3-bit LLM Inference via Interleaved Ternary Quantization with Rotation-Domain Smoothing*. arXiv:2603.27914, 2026.
30. S. Wang et al. *CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs*. arXiv:2606.26650, 2026.
31. C. Yin et al. *TerEffic: Highly Efficient Ternary LLM Inference on FPGA*. arXiv:2502.16473, 2025.
32. L. Dong. *A Geometric Analysis of Sign-Magnitude Asymmetry in a ReLU + RMSNorm Block under Ternary Quantization*. arXiv:2605.18933, 2026.

### 独立評価・コミュニティ

33. Hacker News. *Ternary Bonsai: Top Intelligence at 1.58 Bits* (discussion). https://news.ycombinator.com/item?id=47812749
34. TechCrunch. *PrismML hopes its tiny LLM will change how we all use AI*. 2026-09-17. https://techcrunch.com/2026/09/17/prismml-hopes-its-tiny-llm-could-change-how-we-all-use-ai/
35. ArmanJR. *PrismML-Bonsai-vs-Qwen3.5-Benchmark*. https://github.com/ArmanJR/PrismML-Bonsai-vs-Qwen3.5-Benchmark
36. MiaAI-Lab. *Ternary-Bonsai-27B tool-eval-bench results*. https://github.com/MiaAI-Lab/Ternary-Bonsai-27B-tool-eval-bench-results
37. PrismML-Eng. *Community benchmark: Apple M5 Max*. https://github.com/PrismML-Eng/Bonsai-demo/blob/main/community-benchmarks/ternary-bonsai/metal-m5-max-48gb-macos.md
38. DeepGrove AI. *Bonsai: A Small Ternary-Weight Language Model*. https://github.com/deepgrove-ai/Bonsai

### その他の三値・1-bit系譜

39. A. Kaushal et al. *Spectra: Surprising Effectiveness of Pretraining Ternary Language Models at Scale*. arXiv:2407.12327, 2024.
40. T. Vaidhya et al. *Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models*. arXiv:2506.23025, 2025.
41. T. Chen et al. *TernaryLLM: Ternarized Large Language Model*. arXiv:2406.07177, 2024.
42. H. Shang et al. *PB-LLM: Partially Binarized Large Language Models*. arXiv:2310.00034, 2023.
43. W. Huang et al. *BiLLM: Pushing the Limit of Post-Training Quantization for LLMs*. arXiv:2402.04291, 2024.
44. X. Chen et al. *DB-LLM: Accurate Dual-Binarization for Efficient LLMs*. arXiv:2402.11960, 2024.
45. S. Wang et al. *BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs*. arXiv:2504.18415, 2025.

### ベースモデル

46. Qwen Team. *Qwen3.8-27B*. https://huggingface.co/Qwen/Qwen3.8-27B
47. Qwen Team. *Qwen3.5-2B*. https://huggingface.co/Qwen/Qwen3.5-2B
48. Qwen Team. *Qwen3-8B*. https://huggingface.co/Qwen/Qwen3-8B

### 背景（任意）

49. B. Hassibi, D. G. Stork. *Second Order Derivatives for Network Pruning: Optimal Brain Surgeon*. NeurIPS, 1992.
50. B. Hassibi, A. H. Sayed, T. Kailath. *Linear Estimation*. Prentice Hall, 2000.

---

## 付録D. 本レポートの位置づけ

- 本レポートは、公開情報と独自解析に基づく**独立した技術解説**です。PrismMLの公式見解ではありません。
- モデルの重み・バイナリは再配布していません。解析スクリプトと集計データのみを公開しています。
- ベンチマークの数値は、特に断りのない限りホワイトペーパー/モデルカードの公表値です。独立追試は明記しています。
- 誤りを見つけた場合は、Issueで知らせてください。再現スクリプトとともに検証します。

> **ライセンス**: 本文は CC BY 4.0、コードは MIT。モデルのライセンスはApache 2.0（PrismML）に従います。

---

<p align="center">
  <i>「1.58ビット」の数字だけを見ると、魔法のように思える。バイト列を覗くと、そこには42個のゼロと、86個の符号と、1個のスケールがある。魔法ではなく、設計だ。</i>
</p>