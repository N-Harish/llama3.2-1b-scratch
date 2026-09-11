import torch

from llama3_2_1b_scratch.utils import download_model
from llama3_2_1b_scratch.model import LlamaForCausalLM
from llama3_2_1b_scratch.tokenizer import Llama3Tokenizer


REPO_ID = "Harish241412/llama-3.2-1b-from-scratch"
LOCAL_DIR = "hf_models_downloaded"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("=" * 80)
    print("Hugging Face Download Verification")
    print("=" * 80)
    print(f"Device    : {device}")
    print(f"Dtype     : {dtype}")
    print(f"Repository: {REPO_ID}")
    print(f"Local dir : {LOCAL_DIR}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Download model + tokenizer
    # ------------------------------------------------------------------

    print("\nDownloading model files...")

    local_dir = download_model(
        output_dir=LOCAL_DIR,
        repo_id=REPO_ID,
    )

    print(f"Downloaded to: {local_dir}")

    model_path = local_dir / "llama3_2_1b_scratch_bf16.safetensors"
    tokenizer_path = local_dir / "tokenizer.model"
    tokenizer_config_path = local_dir / "tokenizer_config.json"

    print("\nDownloaded files:")

    for path in (
        model_path,
        tokenizer_path,
        tokenizer_config_path,
    ):
        print(f"  {path} : {'OK' if path.exists() else 'MISSING'}")

    print("\nLoading tokenizer...")

    tokenizer = Llama3Tokenizer(
        tokenizer_path,
        tokenizer_config_path,
    )

    print("Tokenizer loaded successfully.")

    print("\nLoading model from local checkpoint...")

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device=device,
        dtype=dtype,
    )

    print("Model loaded successfully.")

    prompt = "The capital of France is"

    input_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    print(f"\nPrompt    : {prompt}")
    print(f"Input IDs : {input_ids[0].tolist()}")

    # ------------------------------------------------------------------
    # 5. Forward pass
    # ------------------------------------------------------------------

    print("\nRunning forward pass...")

    with torch.inference_mode():
        logits = model(input_ids)

    print(f"Logits shape: {tuple(logits.shape)}")

    next_token_id = logits[:, -1, :].argmax(dim=-1).item()
    next_token = tokenizer.decode([next_token_id])

    print(f"Next token ID : {next_token_id}")
    print(f"Next token    : {next_token!r}")

    assert next_token_id == 12366, (
        f"Unexpected next token ID: {next_token_id}"
    )

    print("Next-token prediction matches expected output.")

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
    # 7. Success
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("HF DOWNLOAD + LOCAL LOAD VERIFICATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()