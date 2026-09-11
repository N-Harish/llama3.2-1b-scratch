import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llama3_2_1b_scratch.model import LlamaForCausalLM
from llama3_2_1b_scratch.load_weights import load_llama_weights
from safetensors.torch import load_file


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

HF_MODEL = "meta-llama/Llama-3.2-1B"
LOCAL_WEIGHTS = "models/Llama-3.2-1B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
# DTYPE = torch.float32

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Python is a programming language used for",
    "Artificial intelligence is",
    "The quick brown fox",
]



print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_WEIGHTS)


print("Loading Hugging Face model...")

hf_model = AutoModelForCausalLM.from_pretrained(
    LOCAL_WEIGHTS,
    dtype=DTYPE,
).to(DEVICE)

hf_model.eval()



print("Loading scratch model...")

scratch_model = LlamaForCausalLM(
    d_model=2048,
    n_layers=16,
    n_q_heads=32,
    n_kv_heads=8,
    d_ff=8192,
    vocab_size=128256,
).to(DEVICE)


print("Loading original HF weights into scratch model...")

load_llama_weights(
    scratch_model,
    f"{LOCAL_WEIGHTS}/model.safetensors"
)


scratch_model.eval()


@torch.no_grad()
def compare_prompt(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(DEVICE)

    print("\n" + "=" * 80)
    print(f"PROMPT: {prompt}")
    print(f"TOKENS: {input_ids.tolist()}")

    hf_outputs = hf_model(
        input_ids=input_ids
    )

    hf_logits = hf_outputs.logits[:, -1, :]

    hf_top_values, hf_top_ids = torch.topk(
        hf_logits,
        k=5,
        dim=-1,
    )

    scratch_logits = scratch_model(
        input_ids
    )
    
    scratch_logits = scratch_logits[:, -1, :]

    scratch_top_values, scratch_top_ids = torch.topk(
        scratch_logits,
        k=5,
        dim=-1,
    )

    max_diff = (
        hf_logits.float() -
        scratch_logits.float()
    ).abs().max().item()

    mean_diff = (
        hf_logits.float() -
        scratch_logits.float()
    ).abs().mean().item()

    hf_top1 = hf_top_ids[0, 0].item()
    scratch_top1 = scratch_top_ids[0, 0].item()

    top1_match = hf_top1 == scratch_top1

    hf_top5 = hf_top_ids[0].tolist()
    scratch_top5 = scratch_top_ids[0].tolist()

    top5_match = hf_top5 == scratch_top5

    hf_prediction = tokenizer.decode([hf_top1])
    scratch_prediction = tokenizer.decode([scratch_top1])

    print("\nTop-5 HF:")
    for token_id, value in zip(
        hf_top_ids[0],
        hf_top_values[0],
    ):
        print(
            f"  {token_id.item():6d} "
            f"{tokenizer.decode([token_id.item()]):20s} "
            f"{value.item():.6f}"
        )

    print("\nTop-5 Scratch:")
    for token_id, value in zip(
        scratch_top_ids[0],
        scratch_top_values[0],
    ):
        print(
            f"  {token_id.item():6d} "
            f"{tokenizer.decode([token_id.item()]):20s} "
            f"{value.item():.6f}"
        )

    print("\nLogit difference:")
    print(f"  Max :  {max_diff}")
    print(f"  Mean:  {mean_diff}")

    print("\nPrediction:")
    print(f"  HF      : {hf_prediction!r}")
    print(f"  Scratch : {scratch_prediction!r}")

    print("\nMATCH:")
    print(f"  Top-1: {top1_match}")
    print(f"  Top-5: {top5_match}")

    return top1_match, top5_match


results = []

for prompt in PROMPTS:
    results.append(
        compare_prompt(prompt)
    )

print("\n")
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

for prompt, (top1, top5) in zip(PROMPTS, results):
    print(
        f"{'PASS' if top1 else 'FAIL'} | "
        f"Top-1={top1} | "
        f"Top-5={top5} | "
        f"{prompt}"
    )

all_top1 = all(x[0] for x in results)
all_top5 = all(x[1] for x in results)

print("\nOverall:")
print(f"  All Top-1 predictions match: {all_top1}")
print(f"  All Top-5 predictions match: {all_top5}")
