from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "Harish241412/llama-3.2-1b-from-scratch"

MODEL_FILENAME = "llama3_2_1b_scratch_bf16.safetensors"
TOKENIZER_FILENAME = "tokenizer.model"
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"


def download_model(
    output_dir: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
) -> Path:
    """
    Download model weights and tokenizer files from Hugging Face.

    Args:
        output_dir: Directory where the files will be saved.
        repo_id: Hugging Face repository ID.

    Returns:
        Path to the output directory.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in (
        MODEL_FILENAME,
        TOKENIZER_FILENAME,
        TOKENIZER_CONFIG_FILENAME,
    ):
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
        )

    return output_dir