import torch

from llama3_2_1b_scratch.model import LlamaForCausalLM
from llama3_2_1b_scratch.tokenizer import Llama3Tokenizer


CHECKPOINT = "hf_models/llama3_2_1b_scratch_bf16.safetensors"
TOKENIZER_MODEL = "hf_models/tokenizer.model"
TOKENIZER_CONFIG = "hf_models/tokenizer_config.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Scratch Checkpoint Verification")
    print("=" * 80)
    print(f"Device     : {device}")
    print(f"Dtype      : torch.bfloat16")
    print(f"Checkpoint : {CHECKPOINT}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------

    tokenizer = Llama3Tokenizer(
        TOKENIZER_MODEL,
        TOKENIZER_CONFIG,
    )

    # ------------------------------------------------------------------
    # Load scratch checkpoint
    # ------------------------------------------------------------------

    print("\nLoading checkpoint...")

    model = LlamaForCausalLM.from_pretrained(
        CHECKPOINT,
        device=device,
        dtype=torch.bfloat16,
        use_kvcache=False,
    )

    print("Checkpoint loaded successfully.")

    # ------------------------------------------------------------------
    # Model information
    # ------------------------------------------------------------------

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters : {num_parameters:,}")

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    prompt = "The capital of France is"

    input_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    print(f"\nPrompt     : {prompt}")
    print(f"Input IDs  : {input_ids.tolist()[0]}")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    print("\nRunning forward pass...")

    with torch.inference_mode():
        logits = model(input_ids)

    print(f"Logits shape: {tuple(logits.shape)}")

    expected_shape = (
        1,
        input_ids.shape[1],
        128256,
    )

    assert logits.shape == expected_shape, (
        f"Unexpected logits shape: "
        f"{tuple(logits.shape)} != {expected_shape}"
    )

    print("Forward pass successful.")

    # ------------------------------------------------------------------
    # Greedy next-token prediction
    # ------------------------------------------------------------------

    next_token_id = logits[:, -1, :].argmax(dim=-1)

    next_token = tokenizer.decode(
        next_token_id.tolist()
    )

    print(f"\nNext token ID : {next_token_id.item()}")
    print(f"Next token    : {repr(next_token)}")

    expected_token_id = 12366

    assert next_token_id.item() == expected_token_id, (
        f"Unexpected next token: "
        f"{next_token_id.item()} != {expected_token_id}"
    )

    print("Next-token prediction matches expected output.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    print("\nRunning generation...")

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=False,
    )

    generated_text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    print(f"Generated IDs  : {generated_ids[0].tolist()}")
    print(f"Generated text : {generated_text}")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CHECKPOINT VERIFICATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()