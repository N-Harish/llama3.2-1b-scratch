# Llama 3.2 1B From Scratch

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Llama%203.2%201B-yellow)](https://huggingface.co/Harish241412/llama-3.2-1b-from-scratch)

A PyTorch implementation of **Meta Llama 3.2 1B from scratch**, including the Transformer architecture, custom tokenizer, KV cache, autoregressive generation, and sampling.

The implementation loads the original Llama 3.2 1B weights into the scratch architecture and is validated against the Hugging Face implementation at both BF16 and FP32 precision.

## Architecture

The model follows the Llama pre-norm Transformer architecture:

<p align="center">
  <img src="https://github.com/user-attachments/assets/177564a8-2a4f-4fc1-829f-3908a1e5c217" width="700" />
</p>

For GQA, the model uses **32 query heads and 8 KV heads**, with four query heads sharing each KV head.


## Model Configuration

| Parameter              |   Value |
| ---------------------- | ------: |
| Parameters             |  ~1.24B |
| Vocabulary size        | 128,256 |
| Hidden dimension       |   2,048 |
| Transformer layers     |      16 |
| Attention heads        |      32 |
| KV heads               |       8 |
| Head dimension         |      64 |
| Intermediate dimension |   8,192 |
| Context length         | 131,072 |
| RMSNorm epsilon        |    1e-5 |
| RoPE base              | 500,000 |
| RoPE scaling factor    |      32 |
| Weight dtype           |    BF16 |

## Features

* Llama 3.2 1B Transformer architecture implemented in PyTorch
* RMSNorm
* Rotary Position Embeddings (RoPE)
* Grouped Query Attention (GQA)
* SwiGLU MLP
* Tied input embeddings / LM head
* Custom Llama 3 tokenizer implementation
* Dynamic KV cache
* Prefill + incremental decode
* Greedy decoding
* Temperature sampling
* Top-k sampling
* Top-p sampling
* Configurable stop tokens
* Original Llama 3.2 1B weight loading
* Hugging Face vs scratch numerical verification

## Verification

### Hugging Face vs Scratch

The scratch implementation was validated against the original Hugging Face Llama 3.2 1B implementation using the same model weights, tokenizer, input prompts, and computation.

The forward pass was compared component-by-component, including:

* Token embeddings
* RMSNorm
* Q / K / V projections
* Q / K / V head reshaping
* Rotary Position Embeddings (RoPE)
* Grouped Query Attention (GQA)
* Attention scores
* Causal masking and softmax
* Attention context
* Output projection
* Residual connections
* SwiGLU MLP
* Final normalization
* LM head / final logits

#### BF16

Under BF16 computation, the final logits matched closely, with a maximum absolute difference of **0.1875**.

For the prompt:

```text
"The capital of France is"
```

both implementations produced the same greedy prediction:

```text
Hugging Face : ' Paris' (ID 12366)
Scratch      : ' Paris' (ID 12366)
```

The **Top-1 and Top-5 predictions were identical**.

#### FP32

For additional numerical verification, the scratch implementation was compared against the Hugging Face implementation in FP32 across five prompts.

| Prompt                                    | Top-1 | Top-5 | Max Absolute Difference |
| ----------------------------------------- | ----- | ----- | ----------------------: |
| The capital of France is                  | Match | Match |                 9.06e-6 |
| The largest planet in our solar system is | Match | Match |                 9.54e-6 |
| Python is a programming language used for | Match | Match |                 1.14e-5 |
| Artificial intelligence is                | Match | Match |                 9.54e-6 |
| The quick brown fox                       | Match | Match |                 1.77e-5 |

**All Top-1 and Top-5 predictions matched**, with the maximum absolute logit difference below **2e-5**.

### KV Cache vs No KV Cache

The KV-cache implementation was independently validated against the non-cached decoding path using the same scratch model and weights.

For each test prompt, validation was performed in two stages:

1. **Prompt prefill**

   * The complete prompt was passed to both paths in a single forward pass.
   * The resulting next-token logits were compared.

2. **Autoregressive decoding**

   * The next token was selected using greedy decoding.
   * In the **non-KV path**, the complete sequence was passed through the model again at every generation step.
   * In the **KV-cache path**, only the newly generated token was passed to the model while previously computed K/V states were reused.
   * At every generation step, logits and selected next tokens were compared.
   * The KV-cache length was checked against the expected sequence length.

Validation was performed across five prompts:

```text
"The capital of France is"

"The largest planet in our solar system is"

"Python is a programming language used for"

"Artificial intelligence is"

"The quick brown fox"
```

Before each independent test, the KV cache was cleared to ensure that no state from a previous sequence was reused.

The KV-cache and non-KV paths produced matching logits and greedy next-token predictions throughout autoregressive generation when evaluated in FP32. Maximum logit differences remained on the order of **1e-5 to 3e-5**, and all generated token sequences remained identical.

This verifies that the KV cache changes **how previously computed attention states are reused**, without changing the model's autoregressive decoding behavior.

> **Note:** When tested in BF16, small numerical differences were observed between the two execution paths. The KV-cache and non-KV paths use different tensor shapes and can therefore trigger different GPU kernels and floating-point reduction orders. Because BF16 has limited mantissa precision, these small differences can occasionally propagate through the model and flip the greedy `argmax` when two logits are very close. FP32 was therefore used to verify functional equivalence of the two decoding paths independently of this precision-sensitive effect.

### `generate()` Method

A public `generate()` method was implemented to provide autoregressive text generation.

The method supports two execution paths:

```text
LlamaForCausalLM(use_kvcache=False)
    → Full-sequence recomputation at every step

LlamaForCausalLM(use_kvcache=True)
    → KV-cache based autoregressive decoding
```

Supported decoding features:

* Greedy decoding
* Temperature sampling
* Top-k sampling
* Top-p sampling
* Top-k + top-p sampling
* Configurable stop token IDs

For the KV-cache path:

1. **Prefill** — process the complete prompt and populate the KV cache.
2. **Decode** — process only the newly generated token.
3. **Cache reuse** — reuse previously computed K/V states.
4. **Token selection** — apply greedy decoding or the configured sampling strategy.

For the non-KV path, the complete generated sequence is recomputed at every step.

The public generation method was verified using the same five prompts used for KV-cache validation. For greedy decoding, all five prompts produced **identical generated token sequences** across the two execution paths.

Sampling was additionally verified across different temperature, top-k, and top-p configurations, including deterministic reproduction with a fixed random seed.

## KV Cache Benchmark

KV caching avoids recomputing the key (K) and value (V) projections for tokens that have already been processed during autoregressive decoding. Instead, the cached K/V states are reused and only the newly generated token is processed.

The benchmark below compares **per-token decode latency** with and without KV caching across different context lengths.

| Context |     No KV |       KV |    Speedup |
| ------: | --------: | -------: | ---------: |
|      16 |  28.84 ms | 20.90 ms |      1.38× |
|      32 |  24.29 ms | 26.97 ms |      0.90× |
|      64 |  27.47 ms | 21.52 ms |      1.28× |
|     128 |  32.22 ms | 21.50 ms |      1.50× |
|     256 |  60.62 ms | 21.39 ms |      2.83× |
|     512 | 119.99 ms | 26.41 ms |      4.54× |
|    1024 | 268.96 ms | 19.44 ms | **13.83×** |

### Observations

* For short contexts, KV caching provides limited benefit and can occasionally be slightly slower due to cache-management and kernel-launch overhead.
* As the context grows, the cost of recomputing the entire sequence at every decode step increases rapidly in the no-KV implementation.
* With KV caching, previously computed K/V states are reused, so each subsequent decode step only computes K/V for the newly generated token.
* At a **1024-token context**, KV caching reduces measured per-token latency from **268.96 ms to 19.44 ms**, corresponding to a **13.83× speedup**.
* The increasing speedup with context length demonstrates the primary advantage of KV caching for autoregressive Transformer inference: **avoiding repeated computation over the existing context**.

> **Note:** These measurements are from the scratch Llama 3.2 1B implementation and are intended to demonstrate the scaling behavior of KV caching. Small-context measurements can be affected substantially by fixed runtime and kernel-launch overhead.

## Tokenizer

A custom Llama 3 tokenizer implementation is included using the original Llama tokenizer vocabulary and BPE data.

It supports:

* Text encoding and decoding
* Batch encoding and decoding
* Special tokens
* Token ↔ ID conversion
* Llama 3 chat formatting

Example:

```python
ids = tokenizer.encode("The capital of France is")
text = tokenizer.decode(ids)
```

The tokenizer was also verified end-to-end with the scratch model:

```text
Text
 ↓
Custom Tokenizer
 ↓
Token IDs
 ↓
Scratch Llama 3.2 1B
 ↓
Generated Token IDs
 ↓
Custom Tokenizer
 ↓
Generated Text
```

For the prompt `"The capital of France is"`, both KV-cache and non-KV generation produced the same token sequence and decoded output.

## Installation

```bash
git clone https://github.com/<username>/llama3.2-1b-scratch.git
cd llama3.2-1b-scratch
uv sync
```

## Usage

### Download model

Download the model checkpoint and tokenizer files from the Hugging Face repository:

```python
from llama3_2_1b_scratch.utils import download_model

local_dir = download_model(
    output_dir="llama3-2-1b",
    repo_id="Harish241412/llama-3.2-1b-from-scratch",
)
```

The following files are downloaded:

* `llama3_2_1b_scratch_bf16.safetensors`
* `tokenizer.model`
* `tokenizer_config.json`

### Load model and tokenizer

```python
import torch

from llama3_2_1b_scratch import LlamaForCausalLM
from llama3_2_1b_scratch.tokenizer import Llama3Tokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlamaForCausalLM.from_pretrained(
    local_dir / "llama3_2_1b_scratch_bf16.safetensors",
    device=device,
    dtype=torch.bfloat16,
    use_kvcache=True,
)

tokenizer = Llama3Tokenizer(
    local_dir / "tokenizer.model",
    local_dir / "tokenizer_config.json",
)
```

### Generate text

```python
prompt = "The capital of France is"

input_ids = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device,
)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=False,
    )

print(tokenizer.decode(output_ids[0].tolist()))
```

Example output:

```text
The capital of France is Paris, and the capital of the United States is
```

### Sampling

The `generate()` method also supports temperature, top-k, and top-p sampling:

```python
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )
```

`from_pretrained()` loads the checkpoint from the local filesystem and does not perform model downloads itself.


## Project Structure

```text
llama3.2-1b-scratch/
├───README.md
├───pyproject.toml
├───scripts
│       benchmark_kv.py
│       save_checkpoint.py
│       verify_checkpoint.py
│       verify_eos.py
│       verify_generate.py
│       verify_hf_download.py
│       verify_hf_vs_scratch.py
│       verify_kv_cache.py
│       verify_llama.py
│       verify_sampling.py
│       verify_tokenizer_model.py
│
├───src
│   └───llama3_2_1b_scratch
│           __init__.py
│           gqa.py
│           kvcache.py
│           load_weights.py
│           model.py
│           rmsnorm.py
│           rope.py
│           swiglu.py
│           tokenizer.py
│           utils.py
│
└───tests
        tokenizer_test.py
```

## Verification Scripts

```bash
uv run .\scripts\verify_llama.py
uv run .\scripts\verify_hf_vs_scratch.py
uv run .\scripts\verify_kv_cache.py
uv run .\scripts\verify_generate.py
uv run .\scripts\verify_sampling.py
uv run .\scripts\verify_eos.py
uv run .\scripts\verify_tokenizer_model.py
uv run .\scripts\benchmark_kv.py
uv run .\scripts\verify_hf_download.py
```

## Future Work

Planned extensions include:

* **Variable-length batched generation** with per-sequence attention masks, position IDs, and KV-cache lengths.
* **Per-sequence EOS handling** so individual sequences can finish without stopping the entire batch.
* **More efficient batch inference** for prompts with different lengths while retaining the current batch-aware KV-cache design.
* **Training from scratch** using the implemented architecture as a foundation for further LLM experimentation.
* **Reasoning and post-training experiments**, including supervised fine-tuning and preference optimization.


## License

The **source code** in this repository is licensed under the **Apache License 2.0**.

The **Llama-derived model weights** are distributed under the **Llama 3.2 Community License** and are not licensed under Apache-2.0.

See [`LICENSE`](LICENSE) for the Apache-2.0 license and [`LICENSE.txt`](LICENSE.txt) / [`NOTICE`](NOTICE) for the applicable Llama license and attribution.

> **Built with Llama**

Required attribution:

```text
Llama 3.2 is licensed under the Llama 3.2 Community License,
Copyright © Meta Platforms, Inc. All Rights Reserved.
```

Users should review the Llama 3.2 Community License and Acceptable Use Policy before using or redistributing the Llama-derived model.
