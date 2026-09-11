import torch
from pathlib import Path
from safetensors.torch import save_file

from llama3_2_1b_scratch.model import LlamaForCausalLM
from llama3_2_1b_scratch.load_weights import load_llama_weights


MODEL_WEIGHTS = "models/Llama-3.2-1B/model.safetensors"
OUTPUT_PATH = "hf_models/llama3_2_1b_scratch_bf16.safetensors"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # Build scratch model
    model = LlamaForCausalLM().to(
        device=device,
        dtype=torch.bfloat16,
    )

    # Load/map original Llama weights
    load_llama_weights(
        model,
        MODEL_WEIGHTS,
    )

    model.eval()

    # Convert checkpoint to BF16 and move tensors to CPU
    state_dict = {
        key: value.detach().to(torch.bfloat16).cpu()
        for key, value in model.state_dict().items()
    }

    print(f"Saving {len(state_dict)} tensors...")

    # Create output directory if it doesn't exist
    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file(
        state_dict,
        OUTPUT_PATH,
        metadata={
            "model_type": "llama3_2_1b_scratch",
            "base_model": "Llama 3.2 1B",
            "dtype": "bfloat16",
        },
    )

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()