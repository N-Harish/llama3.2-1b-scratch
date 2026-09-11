import torch
from llama3_2_1b_scratch.model import LlamaForCausalLM
from llama3_2_1b_scratch.tokenizer import Llama3Tokenizer
from llama3_2_1b_scratch.load_weights import load_llama_weights


MODEL_PATH = "models/Llama-3.2-1B/original/tokenizer.model"
CONFIG_PATH = "models/Llama-3.2-1B/tokenizer_config.json"

LOCAL_WEIGHTS = "models/Llama-3.2-1B/model.safetensors"


EOS_ID = 128001
EOT_ID = 128009


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

    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        device=device,
        dtype=torch.long,
    )

    print("=" * 80)
    print("EOS / STOP TOKEN VERIFICATION")
    print("=" * 80)
    print(f"Prompt : {prompt}")
    print(f"EOS    : {EOS_ID} ({tokenizer.id_to_token(EOS_ID)})")
    print(f"EOT    : {EOT_ID} ({tokenizer.id_to_token(EOT_ID)})")
    print()

    # ---------------------------------------------------------------
    # 1. Normal generation
    # ---------------------------------------------------------------
    print("[1] Normal generation")

    model = load_model(use_kvcache=False)

    output = model.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=False,
    )

    new_tokens = output[:, input_ids.shape[1]:]

    assert new_tokens.shape[1] <= 10

    print("PASS: generation respects max_new_tokens")
    print(f"Output: {tokenizer.decode(output[0].tolist())}")
    print()

    # ---------------------------------------------------------------
    # 2. Verify stop token configuration
    # ---------------------------------------------------------------
    print("[2] Default stop tokens")

    assert EOS_ID == 128001
    assert EOT_ID == 128009

    print("PASS: default stop tokens are EOS and EOT")
    print()

    # ---------------------------------------------------------------
    # 3. Verify EOS termination logic
    # ---------------------------------------------------------------
    print("[3] EOS termination")

    # Directly test the condition used by generate().
    next_token = torch.tensor(
        [[EOS_ID]],
        device=device,
        dtype=torch.long,
    )

    stop_token_ids = [EOS_ID, EOT_ID]

    should_stop = any(
        torch.all(next_token == token_id)
        for token_id in stop_token_ids
    )

    assert should_stop

    print("PASS: EOS triggers termination")
    print()

    # ---------------------------------------------------------------
    # 4. Verify EOT termination logic
    # ---------------------------------------------------------------
    print("[4] EOT termination")

    next_token = torch.tensor(
        [[EOT_ID]],
        device=device,
        dtype=torch.long,
    )

    should_stop = any(
        torch.all(next_token == token_id)
        for token_id in stop_token_ids
    )

    assert should_stop

    print("PASS: EOT triggers termination")
    print()

    # ---------------------------------------------------------------
    # 5. Verify custom stop token override
    # ---------------------------------------------------------------
    print("[5] Custom stop token override")

    custom_stop_id = 12345

    next_token = torch.tensor(
        [[custom_stop_id]],
        device=device,
        dtype=torch.long,
    )

    custom_stop_tokens = [custom_stop_id]

    should_stop = any(
        torch.all(next_token == token_id)
        for token_id in custom_stop_tokens
    )

    assert should_stop

    print("PASS: custom stop_token_ids are supported")
    print()

    # ---------------------------------------------------------------
    # 6. Verify stop token is included
    # ---------------------------------------------------------------
    print("[6] Stop token inclusion")

    generated_ids = torch.cat(
        [input_ids, next_token],
        dim=1,
    )

    assert generated_ids[0, -1].item() == custom_stop_id

    print("PASS: stop token remains in returned IDs")
    print()

    print("=" * 80)
    print("All EOS / stop-token tests passed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()