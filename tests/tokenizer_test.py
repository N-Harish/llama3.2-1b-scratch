from llama3_2_1b_scratch.tokenizer import Llama3Tokenizer


# =================================================================
# TEST
# =================================================================

if __name__ == "__main__":

    tokenizer = Llama3Tokenizer(
        model_path="./models/Llama-3.2-1B/original/tokenizer.model",
        config_path="./models/Llama-3.2-1B/tokenizer_config.json",
    )

    # =============================================================
    # 1. NORMAL TEXT
    # =============================================================

    text = "Hello, world!"

    ids = tokenizer.encode(
        text,
        bos=True
    )

    print("=" * 60)
    print("NORMAL TEXT")
    print("=" * 60)

    print("TEXT:")
    print(text)

    print("\nIDS:")
    print(ids)

    print("\nDECODED:")
    print(tokenizer.decode(ids))

    print("\nDECODED WITHOUT SPECIAL TOKENS:")
    print(
        tokenizer.decode(
            ids,
            skip_special_tokens=True
        )
    )

    # =============================================================
    # 2. SPECIAL TOKENS
    # =============================================================

    print("\n" + "=" * 60)
    print("SPECIAL TOKENS")
    print("=" * 60)

    for token in [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",
        "<|eot_id|>",
        "<|python_tag|>",
    ]:
        print(
            f"{token}: "
            f"{tokenizer.token_to_id(token)}"
        )

    # =============================================================
    # 3. CHAT
    # =============================================================

    messages = [
        {
            "role": "system",
            "content": (
                "You are a pirate chatbot who always "
                "responds in pirate speak!"
            ),
        },
        {
            "role": "user",
            "content": "Who are you?",
        },
    ]

    chat_ids = tokenizer.encode_chat(
        messages,
        add_generation_prompt=True,
    )

    print("\n" + "=" * 60)
    print("CHAT")
    print("=" * 60)

    print("\nCHAT IDS:")
    print(chat_ids)

    print("\nCHAT PROMPT:")
    print(tokenizer.decode(chat_ids))

    from transformers import AutoTokenizer
    LOCAL_WEIGHTS = "meta-llama/Llama-3.2-1B-Instruct"

    hf_tokenizer = AutoTokenizer.from_pretrained(LOCAL_WEIGHTS)

    hf_ids = hf_tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    )

    my_ids = tokenizer.encode_chat(
        messages,
        add_generation_prompt=True,
    )

    print(hf_ids['input_ids'])
    print(my_ids)
    print(hf_ids['input_ids'] == my_ids)