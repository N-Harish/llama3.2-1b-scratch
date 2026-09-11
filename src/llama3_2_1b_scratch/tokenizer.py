import json
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from datetime import datetime


class Llama3Tokenizer:
    """
    Llama 3.2 tokenizer.

    Handles:
        - BPE tokenization
        - Special-token mapping
        - BOS/EOS insertion
        - Decode
        - Optional skipping of special tokens
        - Llama 3.2 Instruct chat formatting
        - Batch encode/decode

    Files:
        tokenizer.model
        tokenizer_config.json
    """

    def __init__(self, model_path, config_path):
        model_path = Path(model_path)
        config_path = Path(config_path)

        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        if not config_path.is_file():
            raise FileNotFoundError(config_path)

        # =========================================================
        # 1. Load TikToken BPE vocabulary
        # =========================================================

        mergeable_ranks = load_tiktoken_bpe(str(model_path))

        # =========================================================
        # 2. Load tokenizer configuration
        # =========================================================

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.config = config

        # Special token string -> ID
        #
        # We get these directly from the model's metadata instead
        # of assuming that reserved tokens have sequential IDs.
        self.special_tokens = {
            info["content"]: int(token_id)
            for token_id, info in config["added_tokens_decoder"].items()
            if info.get("special", False)
        }

        # ID -> special token string
        self.special_tokens_reverse = {
            token_id: token
            for token, token_id in self.special_tokens.items()
        }

        # =========================================================
        # 3. Llama tokenizer regex
        # =========================================================

        pat_str = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
            r"|[^\r\n\p{L}\p{N}]?\p{L}+"
            r"|\p{N}{1,3}"
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
            r"|\s*[\r\n]+"
            r"|\s+(?!\S)"
            r"|\s+"
        )

        # =========================================================
        # 4. Create TikToken Encoding
        # =========================================================

        self.model = tiktoken.Encoding(
            name=model_path.stem,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        # =========================================================
        # 5. BOS / EOS from metadata
        # =========================================================

        self.bos_token = config["bos_token"]
        self.eos_token = config["eos_token"]

        self.bos_id = self.special_tokens[self.bos_token]
        self.eos_id = self.special_tokens[self.eos_token]

    # =============================================================
    # CORE ENCODE
    # =============================================================

    def encode(self, text, bos=False, eos=False):
        """
        Convert normal text -> token IDs.

        Example:

            "Hello, world!"

        ->

            [9906, 11, 1917, 0]

        With BOS:

            [128000, 9906, 11, 1917, 0]
        """

        ids = []

        if bos:
            ids.append(self.bos_id)

        # Do not allow arbitrary text to inject special tokens.
        #
        # Special tokens are inserted explicitly by this class.
        ids.extend(
            self.model.encode(
                text,
                allowed_special=set()
            )
        )

        if eos:
            ids.append(self.eos_id)

        return ids

    # =============================================================
    # CORE DECODE
    # =============================================================

    def decode(self, ids, skip_special_tokens=False):
        """
        Convert token IDs -> text.

        Example:

            [128000, 9906, 11, 1917, 0]

        ->

            "<|begin_of_text|>Hello, world!"

        With skip_special_tokens=True:

            "Hello, world!"
        """

        if not skip_special_tokens:
            return self.model.decode(ids)

        normal_ids = [
            token_id
            for token_id in ids
            if token_id not in self.special_tokens_reverse
        ]

        return self.model.decode(normal_ids)

    # =============================================================
    # SPECIAL TOKENS
    # =============================================================

    def token_to_id(self, token):
        """
        Special token string -> token ID.
        """

        if token not in self.special_tokens:
            raise KeyError(
                f"Unknown special token: {token}"
            )

        return self.special_tokens[token]

    def id_to_token(self, token_id):
        """
        Token ID -> token string.

        For special tokens:
            returns the special token string.

        For normal tokens:
            decodes the token using TikToken.
        """

        if token_id in self.special_tokens_reverse:
            return self.special_tokens_reverse[token_id]

        return self.model.decode([token_id])

    # =============================================================
    # CHAT ENCODE
    # =============================================================

    def encode_chat(
        self,
        messages,
        add_generation_prompt=True,
    ):
        ids = []

        # =========================================================
        # BOS
        # =========================================================

        ids.append(
            self.token_to_id("<|begin_of_text|>")
        )

        # =========================================================
        # Extract system message
        # =========================================================

        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"].strip()
            remaining_messages = messages[1:]
        else:
            system_message = ""
            remaining_messages = messages

        # =========================================================
        # System message
        # =========================================================

        ids.append(
            self.token_to_id("<|start_header_id|>")
        )

        ids.extend(
            self.model.encode(
                "system",
                allowed_special=set()
            )
        )

        ids.append(
            self.token_to_id("<|end_header_id|>")
        )

        today = datetime.now().strftime("%d %b %Y")

        system_text = (
            "\n\n"
            "Cutting Knowledge Date: December 2023\n"
            f"Today Date: {today}\n\n"
            + system_message
        )

        ids.extend(
            self.model.encode(
                system_text,
                allowed_special=set()
            )
        )

        ids.append(
            self.token_to_id("<|eot_id|>")
        )

        # =========================================================
        # Remaining messages
        # =========================================================

        for message in remaining_messages:

            role = message["role"]
            content = message["content"].strip()

            ids.append(
                self.token_to_id("<|start_header_id|>")
            )

            ids.extend(
                self.model.encode(
                    role,
                    allowed_special=set()
                )
            )

            ids.append(
                self.token_to_id("<|end_header_id|>")
            )

            ids.extend(
                self.model.encode(
                    "\n\n" + content,
                    allowed_special=set()
                )
            )

            ids.append(
                self.token_to_id("<|eot_id|>")
            )

        # =========================================================
        # Generation prompt
        # =========================================================

        if add_generation_prompt:

            ids.append(
                self.token_to_id("<|start_header_id|>")
            )

            ids.extend(
                self.model.encode(
                    "assistant",
                    allowed_special=set()
                )
            )

            ids.append(
                self.token_to_id("<|end_header_id|>")
            )

            ids.extend(
                self.model.encode(
                    "\n\n",
                    allowed_special=set()
                )
            )

        return ids

    # =============================================================
    # CHAT DECODE
    # =============================================================

    def decode_chat(self, ids):
        """
        Decode chat token IDs into the readable Llama 3
        formatted prompt.
        """

        return self.decode(ids)

    # =============================================================
    # BATCH ENCODE / DECODE
    # =============================================================

    def encode_batch(self, texts, bos=False, eos=False):
        return [
            self.encode(
                text,
                bos=bos,
                eos=eos
            )
            for text in texts
        ]

    def decode_batch(
        self,
        batch_ids,
        skip_special_tokens=False,
    ):
        return [
            self.decode(
                ids,
                skip_special_tokens=skip_special_tokens
            )
            for ids in batch_ids
        ]
