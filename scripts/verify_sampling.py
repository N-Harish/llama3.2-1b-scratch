import torch
from transformers import AutoTokenizer

from llama3_2_1b_scratch.model import LlamaForCausalLM
from llama3_2_1b_scratch.load_weights import load_llama_weights


MODEL_PATH = "models/Llama-3.2-1B"

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 30


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("=" * 80)
    print("Llama 3.2 1B Sampling Test")
    print("=" * 80)
    print(f"Device : {device}")
    print(f"Dtype  : {dtype}")
    print(f"Prompt : {PROMPT}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = LlamaForCausalLM(
        vocab_size=128256,
        d_model=2048,
        n_layers=16,
        n_q_heads=32,
        n_kv_heads=8,
        d_ff=8192,
        use_kvcache=True,
    ).to(device=device, dtype=dtype)

    load_llama_weights(model, f"{MODEL_PATH}/model.safetensors")

    model.eval()

    input_ids = tokenizer(
        PROMPT,
        return_tensors="pt"
    ).input_ids.to(device)

    # ------------------------------------------------------------------
    # 1. Greedy
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. GREEDY")
    print("=" * 80)

    output = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    print(tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ))

    # ------------------------------------------------------------------
    # 2. Temperature only
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. TEMPERATURE SAMPLING")
    print("=" * 80)

    for temperature in [0.5, 0.7, 1.0, 1.5]:

        torch.manual_seed(42)

        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
        )

        print(f"\nTemperature = {temperature}")
        print(tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ))

    # ------------------------------------------------------------------
    # 3. Top-k
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. TOP-K SAMPLING")
    print("=" * 80)

    for top_k in [1, 10, 50]:

        torch.manual_seed(42)

        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_k=top_k,
        )

        print(f"\nTop-k = {top_k}")
        print(tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ))

    # ------------------------------------------------------------------
    # 4. Top-p
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. TOP-P SAMPLING")
    print("=" * 80)

    for top_p in [0.5, 0.8, 0.9, 0.95]:

        torch.manual_seed(42)

        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=top_p,
        )

        print(f"\nTop-p = {top_p}")
        print(tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ))

    # ------------------------------------------------------------------
    # 5. Top-k + Top-p
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. TOP-K + TOP-P")
    print("=" * 80)

    torch.manual_seed(42)

    output = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )

    print(
        tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
    )

    # ------------------------------------------------------------------
    # 6. Reproducibility
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. REPRODUCIBILITY")
    print("=" * 80)

    torch.manual_seed(123)

    output1 = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )

    torch.manual_seed(123)

    output2 = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )

    identical = torch.equal(output1, output2)

    print(f"Same seed → identical output: {identical}")

    assert identical, "Sampling is not reproducible with the same seed!"

    print("\nSampling tests completed successfully.")


if __name__ == "__main__":
    main()