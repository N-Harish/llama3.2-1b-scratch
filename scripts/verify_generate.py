import torch
from transformers import AutoTokenizer

from llama3_2_1b_scratch.model import LlamaForCausalLM
from llama3_2_1b_scratch.load_weights import load_llama_weights



LOCAL_WEIGHTS = "models/Llama-3.2-1B"


MODEL_PATH = "models/Llama-3.2-1B"
PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Python is a programming language used for",
    "Artificial intelligence is",
    "The quick brown fox",
]

MAX_NEW_TOKENS = 20


@torch.inference_mode()
def verify_prompt(model_no_cache, model_cache, input_ids, tokenizer, max_new_tokens):
    # --------------------------------------------------
    # Reset KV cache
    # --------------------------------------------------
    model_cache.clear_kv_cache()

    no_cache_ids = input_ids.clone()
    cache_ids = input_ids.clone()

    prompt_len = input_ids.shape[1]

    print(f"  Prompt length: {prompt_len}")

    # ==================================================
    # PREFILL
    # ==================================================

    position_ids = torch.arange(
        prompt_len,
        device=input_ids.device,
    ).unsqueeze(0)

    logits_no_cache = model_no_cache(
        input_ids,
        position_ids=position_ids,
    )

    logits_cache = model_cache(
        input_ids,
        position_ids=position_ids,
    )

    # Compare prefill logits
    prefill_diff = (
        logits_no_cache[:, -1, :]
        - logits_cache[:, -1, :]
    ).abs().max().item()

    print(f"  Prefill max diff: {prefill_diff}")

    next_no_cache = logits_no_cache[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    next_cache = logits_cache[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    print(
        f"  First token: "
        f"{tokenizer.decode(next_no_cache[0])!r} / "
        f"{tokenizer.decode(next_cache[0])!r}"
    )

    if not torch.equal(next_no_cache, next_cache):
        print("  FAIL: divergence during prefill")
        return False

    no_cache_ids = torch.cat(
        [no_cache_ids, next_no_cache],
        dim=1,
    )

    cache_ids = torch.cat(
        [cache_ids, next_cache],
        dim=1,
    )

    # ==================================================
    # DECODE
    # ==================================================

    current_token = next_cache
    current_position = prompt_len

    for step in range(1, max_new_tokens):

        # ----------------------------------------------
        # NO KV CACHE
        # ----------------------------------------------

        seq_len = no_cache_ids.shape[1]

        position_ids_no_cache = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        logits_no_cache = model_no_cache(
            no_cache_ids,
            position_ids=position_ids_no_cache,
        )

        next_no_cache = logits_no_cache[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        # ----------------------------------------------
        # KV CACHE
        # ----------------------------------------------

        position_ids_cache = torch.tensor(
            [[current_position]],
            device=input_ids.device,
        )

        logits_cache = model_cache(
            current_token,
            position_ids=position_ids_cache,
        )

        next_cache = logits_cache[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        # ----------------------------------------------
        # Compare logits
        # ----------------------------------------------

        max_diff = (
            logits_no_cache[:, -1, :]
            - logits_cache[:, -1, :]
        ).abs().max().item()

        tokens_match = torch.equal(
            next_no_cache,
            next_cache,
        )

        print(
            f"  Step {step:2d} | "
            f"pos={current_position:2d} | "
            f"logit_diff={max_diff:.6f} | "
            f"token={'PASS' if tokens_match else 'FAIL'} | "
            f"{tokenizer.decode(next_no_cache[0])!r} / "
            f"{tokenizer.decode(next_cache[0])!r}"
        )

        if not tokens_match:
            print("\n  First divergence found.")
            print(
                f"  No-KV token: "
                f"{next_no_cache.item()} "
                f"{tokenizer.decode(next_no_cache[0])!r}"
            )
            print(
                f"  KV token:    "
                f"{next_cache.item()} "
                f"{tokenizer.decode(next_cache[0])!r}"
            )

            return False

        no_cache_ids = torch.cat(
            [no_cache_ids, next_no_cache],
            dim=1,
        )

        cache_ids = torch.cat(
            [cache_ids, next_cache],
            dim=1,
        )

        current_token = next_cache
        current_position += 1

    return torch.equal(
        no_cache_ids,
        cache_ids,
    )

@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    dtype = torch.float32

    # --------------------------------------------------
    # Create both execution modes
    # --------------------------------------------------
    model_no_cache = LlamaForCausalLM(
        use_kvcache=False
    ).to(device=device, dtype=dtype)

    model_cache = LlamaForCausalLM(
        use_kvcache=True
    ).to(device=device, dtype=dtype)

    # --------------------------------------------------
    # Load exactly the same weights into both models
    # --------------------------------------------------
    
    load_llama_weights(model_no_cache, f"{LOCAL_WEIGHTS}/model.safetensors")
    load_llama_weights(model_cache, f"{LOCAL_WEIGHTS}/model.safetensors")

    model_no_cache.eval()
    model_cache.eval()

    print("=" * 80)
    print("GENERATE() VERIFICATION")
    print("=" * 80)

    overall_pass = True

    # for prompt in PROMPTS:
    #     print(f"\nPrompt: {prompt!r}")

    #     # --------------------------------------------------
    #     # Tokenize
    #     # --------------------------------------------------
    #     inputs = tokenizer(
    #         prompt,
    #         return_tensors="pt",
    #         add_special_tokens=True,
    #     )

    #     input_ids = inputs["input_ids"].to(device)

    #     # --------------------------------------------------
    #     # Generate WITHOUT KV cache
    #     # --------------------------------------------------
    #     output_no_cache = model_no_cache.generate(
    #         input_ids,
    #         max_new_tokens=MAX_NEW_TOKENS,
    #     )

    #     # --------------------------------------------------
    #     # Generate WITH KV cache
    #     # --------------------------------------------------
    #     output_cache = model_cache.generate(
    #         input_ids,
    #         max_new_tokens=MAX_NEW_TOKENS,
    #     )

    #     # --------------------------------------------------
    #     # Compare token IDs
    #     # --------------------------------------------------
    #     ids_match = torch.equal(
    #         output_no_cache,
    #         output_cache,
    #     )

    #     # --------------------------------------------------
    #     # Decode
    #     # --------------------------------------------------
    #     text_no_cache = tokenizer.decode(
    #         output_no_cache[0],
    #         skip_special_tokens=True,
    #     )

    #     text_cache = tokenizer.decode(
    #         output_cache[0],
    #         skip_special_tokens=True,
    #     )

    #     text_match = text_no_cache == text_cache

    #     # --------------------------------------------------
    #     # Report
    #     # --------------------------------------------------
    #     print(f"  Output length : {output_no_cache.shape[1]}")
    #     print(f"  Token IDs     : {'PASS' if ids_match else 'FAIL'}")
    #     print(f"  Text          : {'PASS' if text_match else 'FAIL'}")

    #     if not ids_match:
    #         print("\n  No-KV IDs:")
    #         print(output_no_cache[0].tolist())

    #         print("\n  KV IDs:")
    #         print(output_cache[0].tolist())

    #     print(f"\n  No-KV: {text_no_cache!r}")
    #     print(f"  KV   : {text_cache!r}")

    #     passed = ids_match and text_match

    #     print(
    #         f"\n  Result: {'PASS' if passed else 'FAIL'}"
    #     )

    #     overall_pass &= passed

    # print("\n" + "=" * 80)
    # print(
    #     f"Overall generate() verification: "
    #     f"{'PASS' if overall_pass else 'FAIL'}"
    # )
    # print("=" * 80)

    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt!r}")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = inputs["input_ids"].to(device)

        passed = verify_prompt(
            model_no_cache,
            model_cache,
            input_ids,
            tokenizer,
            MAX_NEW_TOKENS,
        )

        print(
            f"\n  Result: {'PASS' if passed else 'FAIL'}"
        )

        overall_pass &= passed

    print("\n" + "=" * 80)
    print(
        f"Overall generate() verification: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )
    print("=" * 80)

if __name__ == "__main__":
    main()