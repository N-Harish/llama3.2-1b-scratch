import torch
from llama3_2_1b_scratch.model import LlamaForCausalLM
from llama3_2_1b_scratch.tokenizer import Llama3Tokenizer
from llama3_2_1b_scratch.load_weights import load_llama_weights


MODEL_PATH = "models/Llama-3.2-1B/original/tokenizer.model"
CONFIG_PATH = "models/Llama-3.2-1B/tokenizer_config.json"

LOCAL_WEIGHTS = "models/Llama-3.2-1B/model.safetensors"


def load_model(use_kvcache=False):
    model = LlamaForCausalLM(
        use_kvcache=use_kvcache
    ).to("cuda", dtype=torch.bfloat16)

    load_llama_weights(model, LOCAL_WEIGHTS)

    model.eval()
    return model


def main():
    device = "cuda"

    tokenizer = Llama3Tokenizer(
        MODEL_PATH,
        CONFIG_PATH,
    )

    prompt = "The capital of France is"

    print("=" * 80)
    print("TOKENIZER + SCRATCH MODEL VERIFICATION")
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print()

    # ------------------------------------------------------------------
    # 1. Encode
    # ------------------------------------------------------------------
    print("[1] Tokenizer encode")

    token_ids = tokenizer.encode(prompt)

    assert len(token_ids) > 0
    assert all(isinstance(token_id, int) for token_id in token_ids)

    input_ids = torch.tensor(
        [token_ids],
        device=device,
        dtype=torch.long,
    )

    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    print("PASS: text successfully encoded")
    print()

    # ------------------------------------------------------------------
    # 2. Decode encoded tokens
    # ------------------------------------------------------------------
    print("[2] Tokenizer decode")

    decoded = tokenizer.decode(token_ids)

    assert decoded == prompt

    print(f"Decoded: {decoded!r}")
    print("PASS: encode/decode round-trip")
    print()

    # ------------------------------------------------------------------
    # 3. Scratch model + tokenizer, no KV cache
    # ------------------------------------------------------------------
    print("[3] Scratch model + tokenizer (no KV cache)")

    model_no_kv = load_model(use_kvcache=False)

    output_no_kv = model_no_kv.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=False,
    )

    assert output_no_kv.shape[0] == 1
    assert output_no_kv.shape[1] <= (
        input_ids.shape[1] + 10
    )

    generated_text_no_kv = tokenizer.decode(
        output_no_kv[0].tolist()
    )

    assert generated_text_no_kv.startswith(prompt)

    print(f"Generated IDs: {output_no_kv[0].tolist()}")
    print(f"Generated text: {generated_text_no_kv}")
    print("PASS: tokenizer → scratch model → tokenizer")
    print()

    # ------------------------------------------------------------------
    # 4. Scratch model + tokenizer, KV cache
    # ------------------------------------------------------------------
    print("[4] Scratch model + tokenizer (KV cache)")

    model_kv = load_model(use_kvcache=True)

    output_kv = model_kv.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=False,
    )

    assert output_kv.shape[0] == 1
    assert output_kv.shape[1] <= (
        input_ids.shape[1] + 10
    )

    generated_text_kv = tokenizer.decode(
        output_kv[0].tolist()
    )

    assert generated_text_kv.startswith(prompt)

    print(f"Generated IDs: {output_kv[0].tolist()}")
    print(f"Generated text: {generated_text_kv}")
    print("PASS: tokenizer → scratch model + KV cache → tokenizer")
    print()

    # ------------------------------------------------------------------
    # 5. Compare KV and no-KV outputs
    # ------------------------------------------------------------------
    print("[5] KV vs no-KV end-to-end output")

    assert torch.equal(
        output_no_kv,
        output_kv,
    ), (
        "KV and no-KV generation produced different token IDs.\n"
        f"No-KV: {output_no_kv[0].tolist()}\n"
        f"KV   : {output_kv[0].tolist()}"
    )

    assert generated_text_no_kv == generated_text_kv

    print("PASS: KV and no-KV generated identical token IDs")
    print()

    # ------------------------------------------------------------------
    # 6. Verify generated IDs can be decoded independently
    # ------------------------------------------------------------------
    print("[6] Generated token IDs → text")

    generated_ids = output_kv[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    assert isinstance(generated_text, str)
    assert len(generated_text) > 0

    print(f"Final text: {generated_text}")
    print("PASS: generated token IDs successfully decoded")
    print()

    print("=" * 80)
    print("All tokenizer + scratch model tests passed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()