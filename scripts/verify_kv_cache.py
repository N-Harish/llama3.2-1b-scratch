
import torch
from transformers import AutoTokenizer

from llama3_2_1b_scratch.model import LlamaForCausalLM
from llama3_2_1b_scratch.load_weights import load_llama_weights


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

LOCAL_WEIGHTS = "models/Llama-3.2-1B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Python is a programming language used for",
    "Artificial intelligence is",
    "The quick brown fox",
]

MAX_NEW_TOKENS = 10


# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_WEIGHTS
)


# ---------------------------------------------------------
# Create scratch model
# ---------------------------------------------------------

def create_model(use_kvcache):

    model = LlamaForCausalLM(
        d_model=2048,
        n_layers=16,
        n_q_heads=32,
        n_kv_heads=8,
        d_ff=8192,
        vocab_size=128256,
        use_kvcache=use_kvcache,
    ).to(DEVICE)

    print(
        f"Loading weights "
        f"(KV cache={'ON' if use_kvcache else 'OFF'})..."
    )

    load_llama_weights(
        model,
        f"{LOCAL_WEIGHTS}/model.safetensors"
    )

    model.eval()

    return model


print("\nLoading scratch model without KV cache...")

scratch_no_cache = create_model(
    use_kvcache=False
)


print("\nLoading scratch model with KV cache...")

scratch_with_cache = create_model(
    use_kvcache=True
)


print("\nBoth models loaded.")


# ---------------------------------------------------------
# Compare generation
# ---------------------------------------------------------

@torch.no_grad()
def compare_generation(prompt):

    print("\n" + "=" * 80)
    print(f"PROMPT: {prompt}")
    print("=" * 80)

    # -----------------------------------------------------
    # Tokenize prompt
    # -----------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    prompt_ids = inputs["input_ids"].to(DEVICE)

    prompt_len = prompt_ids.shape[1]

    print(f"Prompt tokens: {prompt_ids.tolist()}")
    print(f"Prompt length: {prompt_len}")

    # -----------------------------------------------------
    # Clear KV cache
    #
    # Important: every prompt gets a fresh cache.
    # -----------------------------------------------------

    scratch_with_cache.clear_kv_cache()

    # -----------------------------------------------------
    # PREFILL
    # -----------------------------------------------------

    no_cache_logits = scratch_no_cache(
        prompt_ids
    )

    cache_logits = scratch_with_cache(
        prompt_ids
    )

    # -----------------------------------------------------
    # Compare prefill logits
    # -----------------------------------------------------

    no_cache_next_logits = (
        no_cache_logits[:, -1, :]
    )

    cache_next_logits = (
        cache_logits[:, -1, :]
    )

    prefill_diff = (
        no_cache_next_logits.float()
        - cache_next_logits.float()
    ).abs()

    prefill_max_diff = prefill_diff.max().item()
    prefill_mean_diff = prefill_diff.mean().item()

    print("\nPrefill comparison:")
    print(f"  Max difference : {prefill_max_diff}")
    print(f"  Mean difference: {prefill_mean_diff}")

    # -----------------------------------------------------
    # First generated token
    # -----------------------------------------------------

    next_token_no_cache = torch.argmax(
        no_cache_next_logits,
        dim=-1,
        keepdim=True,
    )

    next_token_cache = torch.argmax(
        cache_next_logits,
        dim=-1,
        keepdim=True,
    )

    first_token_match = torch.equal(
        next_token_no_cache,
        next_token_cache,
    )

    print("\nFirst generated token:")

    print(
        f"  No-cache: "
        f"{next_token_no_cache.item()} "
        f"{tokenizer.decode([next_token_no_cache.item()])!r}"
    )

    print(
        f"  KV-cache: "
        f"{next_token_cache.item()} "
        f"{tokenizer.decode([next_token_cache.item()])!r}"
    )

    print(
        f"  Match: {first_token_match}"
    )

    # -----------------------------------------------------
    # Generation state
    # -----------------------------------------------------

    no_cache_ids = prompt_ids.clone()
    cache_ids = prompt_ids.clone()

    generated_no_cache = []
    generated_cache = []

    current_token_no_cache = next_token_no_cache
    current_token_cache = next_token_cache

    # The first generated token is at this position.
    current_position = prompt_len

    all_match = first_token_match

    # -----------------------------------------------------
    # Decode
    # -----------------------------------------------------

    for step in range(MAX_NEW_TOKENS):

        # =================================================
        # Append the previously predicted token
        # =================================================

        no_cache_ids = torch.cat(
            [
                no_cache_ids,
                current_token_no_cache,
            ],
            dim=1,
        )

        cache_ids = torch.cat(
            [
                cache_ids,
                current_token_cache,
            ],
            dim=1,
        )

        generated_no_cache.append(
            current_token_no_cache.item()
        )

        generated_cache.append(
            current_token_cache.item()
        )

        # =================================================
        # NO KV CACHE
        #
        # Re-process the complete sequence.
        # =================================================

        no_cache_logits = scratch_no_cache(
            no_cache_ids
        )

        no_cache_next_logits = (
            no_cache_logits[:, -1, :]
        )

        next_token_no_cache = torch.argmax(
            no_cache_next_logits,
            dim=-1,
            keepdim=True,
        )

        # =================================================
        # KV CACHE
        #
        # Only process the newly added token.
        # =================================================

        position_ids = torch.tensor(
            [[current_position]],
            device=DEVICE,
            dtype=torch.long,
        )

        cache_logits = scratch_with_cache(
            current_token_cache,
            position_ids=position_ids,
        )

        cache_next_logits = (
            cache_logits[:, -1, :]
        )

        next_token_cache = torch.argmax(
            cache_next_logits,
            dim=-1,
            keepdim=True,
        )

        # =================================================
        # Compare logits
        # =================================================

        logits_diff = (
            no_cache_next_logits.float()
            - cache_next_logits.float()
        ).abs()

        max_diff = logits_diff.max().item()
        mean_diff = logits_diff.mean().item()

        # =================================================
        # Compare next token
        # =================================================

        token_match = torch.equal(
            next_token_no_cache,
            next_token_cache,
        )

        if not token_match:
            all_match = False

        no_cache_token_id = (
            next_token_no_cache.item()
        )

        cache_token_id = (
            next_token_cache.item()
        )

        no_cache_text = tokenizer.decode(
            [no_cache_token_id]
        )

        cache_text = tokenizer.decode(
            [cache_token_id]
        )

        print(f"\nStep {step + 1}:")

        print(
            f"  Input position : "
            f"{current_position}"
        )

        print(
            f"  No-cache next  : "
            f"{no_cache_token_id} "
            f"{no_cache_text!r}"
        )

        print(
            f"  KV-cache next  : "
            f"{cache_token_id} "
            f"{cache_text!r}"
        )

        print(
            f"  Token match    : "
            f"{token_match}"
        )

        print(
            f"  Logit max diff : "
            f"{max_diff}"
        )

        print(
            f"  Logit mean diff: "
            f"{mean_diff}"
        )

        # =================================================
        # Verify cache length
        # =================================================

        expected_cache_length = (
            current_position + 1
        )

        actual_cache_length = (
            scratch_with_cache
            .model
            .kvcaches[0]
            .seen_tokens
        )

        cache_length_match = (
            actual_cache_length
            == expected_cache_length
        )

        print(
            f"  Cache length   : "
            f"{actual_cache_length} "
            f"(expected {expected_cache_length}) "
            f"-> {cache_length_match}"
        )

        if not cache_length_match:
            all_match = False

        # =================================================
        # Move to next token
        # =================================================

        current_token_no_cache = (
            next_token_no_cache
        )

        current_token_cache = (
            next_token_cache
        )

        current_position += 1

    # -----------------------------------------------------
    # Add final generated token
    # -----------------------------------------------------

    generated_no_cache.append(
        current_token_no_cache.item()
    )

    generated_cache.append(
        current_token_cache.item()
    )

    # -----------------------------------------------------
    # Compare complete generated sequences
    # -----------------------------------------------------

    sequence_match = (
        generated_no_cache
        == generated_cache
    )

    # -----------------------------------------------------
    # Decode final text
    # -----------------------------------------------------

    no_cache_text = tokenizer.decode(
        prompt_ids[0].tolist()
        + generated_no_cache
    )

    cache_text = tokenizer.decode(
        prompt_ids[0].tolist()
        + generated_cache
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "-" * 80)
    print("FINAL RESULT")
    print("-" * 80)

    print(
        f"\nNo-cache generation:"
        f"\n  {no_cache_text!r}"
    )

    print(
        f"\nKV-cache generation:"
        f"\n  {cache_text!r}"
    )

    print(
        f"\nGenerated tokens match: "
        f"{sequence_match}"
    )

    passed = (
        all_match
        and sequence_match
    )

    print(
        f"All checks passed: "
        f"{passed}"
    )

    # -----------------------------------------------------
    # First mismatch
    # -----------------------------------------------------

    if not sequence_match:

        for i, (no_kv, kv) in enumerate(
            zip(
                generated_no_cache,
                generated_cache,
            )
        ):

            if no_kv != kv:

                print(
                    f"\nFirst generated-token mismatch "
                    f"at token {i}:"
                )

                print(
                    f"  No-cache: {no_kv} "
                    f"{tokenizer.decode([no_kv])!r}"
                )

                print(
                    f"  KV-cache: {kv} "
                    f"{tokenizer.decode([kv])!r}"
                )

                break

    return passed


# ---------------------------------------------------------
# Run tests
# ---------------------------------------------------------

results = []

for prompt in PROMPTS:

    passed = compare_generation(
        prompt
    )

    results.append(
        (prompt, passed)
    )


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("KV CACHE CORRECTNESS SUMMARY")
print("=" * 80)

for prompt, passed in results:

    print(
        f"{'PASS' if passed else 'FAIL'}  "
        f"{prompt!r}"
    )

all_passed = all(
    passed
    for _, passed in results
)

print("\n" + "-" * 80)

print(
    f"Overall KV cache correctness: "
    f"{'PASS' if all_passed else 'FAIL'}"
)

print("=" * 80)