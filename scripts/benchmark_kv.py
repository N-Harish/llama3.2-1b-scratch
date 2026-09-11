import argparse
import time

import torch
import torch.nn.functional as F

from llama3_2_1b_scratch.model import LlamaForCausalLM
from llama3_2_1b_scratch.load_weights import load_llama_weights


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

VOCAB_SIZE = 128256
D_MODEL = 2048
N_LAYERS = 16
N_Q_HEADS = 32
N_KV_HEADS = 8
D_FF = 8192

LOCAL_WEIGHTS = "models/Llama-3.2-1B"

WARMUP = 5
ITERATIONS = 20


# ============================================================
# CUDA SYNC
# ============================================================

def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ============================================================
# BENCHMARK HELPER
# ============================================================

def benchmark(fn, iterations=ITERATIONS):

    # Warmup
    for _ in range(WARMUP):
        fn()

    sync()

    times = []

    for _ in range(iterations):
        sync()
        start = time.perf_counter()

        fn()

        sync()
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return (
        sum(times) / len(times),
        min(times),
        max(times),
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(use_kv):

    model = LlamaForCausalLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_q_heads=N_Q_HEADS,
        n_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        use_kvcache=use_kv,
    )

    load_llama_weights(
        model,
        f"{LOCAL_WEIGHTS}/model.safetensors",
    )

    model = model.to(
        device=DEVICE,
        dtype=DTYPE,
    )

    model.eval()

    return model


# ============================================================
# POSITION IDS
# ============================================================

def make_positions(length):
    return torch.arange(
        length,
        device=DEVICE,
        dtype=torch.long,
    ).unsqueeze(0)


# ============================================================
# TRUE KV DECODE
# ============================================================

@torch.no_grad()
def benchmark_kv_decode(model, input_ids):

    B, prompt_len = input_ids.shape

    # --------------------------------------------------------
    # PREFILL
    # --------------------------------------------------------

    model.clear_kv_cache()

    prompt_positions = make_positions(prompt_len)

    sync()
    start = time.perf_counter()

    _ = model(
        input_ids,
        position_ids=prompt_positions,
        mask=True,
    )

    sync()
    end = time.perf_counter()

    prefill_ms = (end - start) * 1000

    # --------------------------------------------------------
    # SINGLE TOKEN
    # --------------------------------------------------------

    next_token = input_ids[:, -1:]

    decode_position = torch.tensor(
        [[prompt_len]],
        device=DEVICE,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # FULL MODEL DECODE
    # --------------------------------------------------------

    def run_decode():

        return model(
            next_token,
            position_ids=decode_position,
            mask=True,
        )

    avg_ms, min_ms, max_ms = benchmark(
        run_decode
    )

    return prefill_ms, avg_ms, min_ms, max_ms


# ============================================================
# TRUE NO-KV DECODE
# ============================================================

@torch.no_grad()
def benchmark_no_kv_decode(model, input_ids):

    prompt_len = input_ids.shape[1]

    # --------------------------------------------------------
    # REAL DECODE CONTEXT
    #
    # Without KV cache, the model has to receive the entire
    # context again to calculate the next-token logits.
    # --------------------------------------------------------

    position_ids = make_positions(prompt_len)

    def run_decode():

        logits = model(
            input_ids,
            position_ids=position_ids,
            mask=True,
        )

        # Only the final token's logits are needed.
        return logits[:, -1, :]

    avg_ms, min_ms, max_ms = benchmark(
        run_decode
    )

    return avg_ms, min_ms, max_ms


# ============================================================
# WHOLE-LAYER KV BREAKDOWN
# ============================================================

@torch.no_grad()
def benchmark_layer_breakdown(model, input_ids):

    prompt_len = input_ids.shape[1]

    # --------------------------------------------------------
    # PREFILL
    # --------------------------------------------------------

    model.clear_kv_cache()

    prompt_positions = make_positions(prompt_len)

    _ = model(
        input_ids,
        position_ids=prompt_positions,
        mask=True,
    )

    # --------------------------------------------------------
    # SINGLE TOKEN
    # --------------------------------------------------------

    next_token = input_ids[:, -1:]

    position_ids = torch.tensor(
        [[prompt_len]],
        device=DEVICE,
        dtype=torch.long,
    )

    # Embedding
    x = model.model.embed_tokens(next_token)

    layer_times = []

    # --------------------------------------------------------
    # EACH COMPLETE TRANSFORMER LAYER
    # --------------------------------------------------------

    for layer_idx, layer in enumerate(model.model.layers):

        cache = model.model.kvcaches[layer_idx]

        # Save the cache position.
        #
        # The layer benchmark below will append one token.
        # Restore it afterwards so each measurement starts
        # from the same cache state.
        original_seen_tokens = cache.seen_tokens

        def run_layer():

            residual = x

            x_norm = layer.input_layernorm(x)

            attn_output = layer.self_attn(
                x_norm,
                position_ids=position_ids,
                mask=True,
                kvcache=cache,
            )

            x_out = residual + attn_output

            residual = x_out

            x_norm = layer.post_attention_layernorm(x_out)

            mlp_output = layer.mlp(x_norm)

            return residual + mlp_output

        # ----------------------------------------------------
        # IMPORTANT:
        # Each benchmark call modifies the KV cache.
        # Restore it after every measurement.
        # ----------------------------------------------------

        def run_layer_with_cache_reset():

            cache.seen_tokens = original_seen_tokens

            return run_layer()

        avg_ms, min_ms, max_ms = benchmark(
            run_layer_with_cache_reset
        )

        # Restore cache once more.
        cache.seen_tokens = original_seen_tokens

        layer_times.append(
            (layer_idx, avg_ms, min_ms, max_ms)
        )

        # Advance x ONCE so that the next layer gets the
        # correct hidden state.
        x = run_layer()

        # run_layer appended one token to the cache.
        # Keep that state because subsequent layers need
        # their normal decode cache state.

    # --------------------------------------------------------
    # FINAL RMSNORM
    # --------------------------------------------------------

    def run_final_norm():
        return model.model.norm(x)

    final_norm_ms, _, _ = benchmark(
        run_final_norm
    )

    hidden = model.model.norm(x)

    # --------------------------------------------------------
    # LM HEAD
    # --------------------------------------------------------

    def run_lm_head():

        return F.linear(
            hidden,
            model.model.embed_tokens.weight,
        )

    lm_head_ms, lm_head_min, lm_head_max = benchmark(
        run_lm_head
    )

    return (
        layer_times,
        final_norm_ms,
        lm_head_ms,
        lm_head_min,
        lm_head_max,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kv",
        action="store_true",
        help="Enable KV cache",
    )

    parser.add_argument(
        "--prompt-len",
        type=int,
        default=256,
        help="Prompt/context length",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Llama 3.2 1B Decode Benchmark")
    print("=" * 80)

    print(f"Device        : {DEVICE}")
    print(f"Dtype         : {DTYPE}")
    print(f"KV cache      : {args.kv}")
    print(f"Prompt length : {args.prompt_len}")
    print(f"Warmup        : {WARMUP}")
    print(f"Iterations    : {ITERATIONS}")

    print("=" * 80)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    model = load_model(
        use_kv=args.kv
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (1, args.prompt_len),
        device=DEVICE,
        dtype=torch.long,
    )

    # ========================================================
    # KV ENABLED
    # ========================================================

    if args.kv:

        prefill_ms, avg_ms, min_ms, max_ms = (
            benchmark_kv_decode(
                model,
                input_ids,
            )
        )

        print()
        print("=" * 80)
        print("KV-CACHE DECODE")
        print("=" * 80)

        print(f"Prefill       : {prefill_ms:8.3f} ms")
        print(f"Decode avg    : {avg_ms:8.3f} ms")
        print(f"Decode min    : {min_ms:8.3f} ms")
        print(f"Decode max    : {max_ms:8.3f} ms")
        print(f"Tokens/sec    : {1000 / avg_ms:8.3f}")

        print("=" * 80)

        # ----------------------------------------------------
        # LAYER BREAKDOWN
        # ----------------------------------------------------

        (
            layer_times,
            final_norm_ms,
            lm_head_ms,
            lm_head_min,
            lm_head_max,
        ) = benchmark_layer_breakdown(
            model,
            input_ids,
        )

        print()
        print("=" * 80)
        print("WHOLE-LAYER DECODE BREAKDOWN")
        print("=" * 80)

        print(
            f"{'Layer':>8}"
            f"{'Avg ms':>12}"
            f"{'Min ms':>12}"
            f"{'Max ms':>12}"
        )

        print("-" * 80)

        total_layers = 0.0

        for layer_idx, avg, min_ms, max_ms in layer_times:

            print(
                f"{layer_idx:>8}"
                f"{avg:>12.3f}"
                f"{min_ms:>12.3f}"
                f"{max_ms:>12.3f}"
            )

            total_layers += avg

        print("-" * 80)

        print(
            f"{'TOTAL':>8}"
            f"{total_layers:>12.3f}"
        )

        print()
        print(f"Final RMSNorm : {final_norm_ms:8.3f} ms")
        print(
            f"LM Head       : {lm_head_ms:8.3f} ms "
            f"(min {lm_head_min:.3f}, max {lm_head_max:.3f})"
        )

        print("=" * 80)

    # ========================================================
    # KV DISABLED
    # ========================================================

    else:

        avg_ms, min_ms, max_ms = (
            benchmark_no_kv_decode(
                model,
                input_ids,
            )
        )

        print()
        print("=" * 80)
        print("NO-KV DECODE")
        print("=" * 80)

        print(
            f"Full-context decode avg : {avg_ms:8.3f} ms"
        )

        print(
            f"Full-context decode min : {min_ms:8.3f} ms"
        )

        print(
            f"Full-context decode max : {max_ms:8.3f} ms"
        )

        print(
            f"Tokens/sec              : {1000 / avg_ms:8.3f}"
        )

        print("=" * 80)


if __name__ == "__main__":
    main()