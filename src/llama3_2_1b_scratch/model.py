from torch import nn
from .rope import LlamaRoPE
from .rmsnorm import RMSNorm
from .gqa import LlamaGQA
from .swiglu import SwiGLU
import torch.nn.functional as F
from .kvcache import KVCache
import torch
from pathlib import Path
from safetensors.torch import load_file


def _sample_next_token(logits, temperature, top_p, top_k):
    if temperature <= 0:
        raise ValueError(
            f"temperature must be > 0 when do_sample=True. "
            f"You passed temperature={temperature}"
        )
    
    logits = logits / temperature
    
    if top_k is not None:
        if top_k <= 0:
            raise ValueError(
                f"top_k must be > 0. You passed top_k={top_k}"
            )
        
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k, dim=-1)
        kth_value = values[..., -1, None]
        logits = logits.masked_fill(logits < kth_value, float("-inf"))

    # Top-p
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError(
                f"top_p must be in (0, 1]. You passed top_p={top_p}"
            )

        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )

        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove, float("-inf")
        )

        logits = torch.zeros_like(logits).scatter(
            -1, sorted_indices, sorted_logits
        )

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class LlamaTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model=2048,
        n_q_heads=32,
        n_kv_heads=8,
        d_ff=8192,
        rope=None,
    ):
        super().__init__()

        self.input_layernorm = RMSNorm(d_model, eps=1e-5)

        self.self_attn = LlamaGQA(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            rope=rope,
        )

        self.post_attention_layernorm = RMSNorm(
            d_model,
            eps=1e-5
        )

        self.mlp = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
        )

    def forward(self, x, position_ids=None, mask=True, kvcache=None):
        residual = x

        x = self.input_layernorm(x)
        x = self.self_attn(
            x,
            position_ids=position_ids,
            mask=mask,
            kvcache=kvcache
        )

        x = residual + x
        residual = x

        x = self.post_attention_layernorm(x)
        x = self.mlp(x)

        x = residual + x

        return x


class LlamaModel(nn.Module):
    def __init__(
        self,
        vocab_size=128256,
        d_model=2048,
        n_layers=16,
        n_q_heads=32,
        n_kv_heads=8,
        d_ff=8192,
        use_kvcache=False,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            vocab_size,
            d_model
        )
        rope = LlamaRoPE()
        self.layers = nn.ModuleList([
            LlamaTransformerBlock(
                d_model=d_model,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                d_ff=d_ff,
                rope=rope,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model, eps=1e-5)

        self.use_kvcache = use_kvcache
        if use_kvcache:
            self.kvcaches = [KVCache() for _ in range(n_layers)]
        else:
            self.kvcaches = None

    def clear_kv_cache(self):
        if self.kvcaches is not None:
            for cache in self.kvcaches:
                cache.clear()

    def forward(self, x, position_ids=None, mask=True):
        x = self.embed_tokens(x)

        for num_layer, layer in enumerate(self.layers):
            kv_cache = (
                self.kvcaches[num_layer] if self.use_kvcache else None
            )

            x = layer(
                x,
                position_ids = position_ids,
                mask = mask,
                kvcache = kv_cache
            )

        x = self.norm(x)

        return x


class LlamaForCausalLM(nn.Module):
    DEFAULT_STOP_TOKEN_IDS = [128001, 128009]

    def __init__(self, vocab_size=128256,
            d_model=2048,
            n_layers=16,
            n_q_heads=32,
            n_kv_heads=8,
            d_ff=8192,
            use_kvcache=False):
        super().__init__()

        self.model = LlamaModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            use_kvcache=use_kvcache
        )
        self.use_kvcache = use_kvcache

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path,
        device=None,
        dtype=torch.bfloat16,
        use_kvcache=False,
    ):
        checkpoint_path = Path(checkpoint_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = cls(
            vocab_size=128256,
            d_model=2048,
            n_layers=16,
            n_q_heads=32,
            n_kv_heads=8,
            d_ff=8192,
            use_kvcache=use_kvcache,
        )

        state_dict = load_file(
            checkpoint_path,
            device=device,
        )

        model.load_state_dict(state_dict, strict=True)

        model.to(device=device, dtype=dtype)
        model.eval()

        return model

    def clear_kv_cache(self):
        if self.use_kvcache:
            self.model.clear_kv_cache()

    def forward(self, input_ids, position_ids=None, mask=True):
        hidden_states = self.model(
            input_ids,
            position_ids=position_ids,
            mask=mask
        )

        logits = F.linear(
            hidden_states,
            self.model.embed_tokens.weight
        )

        return logits

    def _generate_without_kv_cache(self, input_ids, max_new_tokens, do_sample, temperature, top_p, top_k, stop_token_ids=None):
        generated_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            seq_len = generated_ids.shape[1]
            position_ids = torch.arange(
                seq_len,
                device=input_ids.device
            ).unsqueeze(0)
            logits = self(
                generated_ids,
                position_ids=position_ids
            )
            if do_sample:
                next_token = _sample_next_token(logits[:, -1, :], temperature, top_p, top_k)
            else:
                next_token = logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True
                )
            
            
            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=1
            )

            # Then check whether generation should stop.
            if stop_token_ids is not None:
                if any(
                    torch.all(next_token == token_id)
                    for token_id in stop_token_ids
                ):
                    break

        return generated_ids

    def _generate_with_kv_cache(self, input_ids, max_new_tokens, do_sample, temperature, top_p, top_k, stop_token_ids=None):
        self.clear_kv_cache()
        generated_ids = input_ids.clone()
        prompt_len = input_ids.shape[1]

        # -------------------------
        # Prefill
        # -------------------------
        position_ids = torch.arange(
            prompt_len,
            device=input_ids.device
        ).unsqueeze(0)

        logits = self(
            input_ids,
            position_ids=position_ids
        )

        if do_sample:
            next_token = _sample_next_token(logits[:, -1, :], temperature, top_p, top_k)
        else:
            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True
            )

        generated_ids = torch.cat(
            [generated_ids, next_token],
            dim=1
        )

        # If EOS/EOT was generated during prefill,
        # return immediately.
        if stop_token_ids is not None:
            if any(
                torch.all(next_token == token_id)
                for token_id in stop_token_ids
            ):
                return generated_ids

        # -------------------------
        # Decode
        # -------------------------
        for step in range(1, max_new_tokens):

            position_id = prompt_len + step - 1

            position_ids = torch.tensor(
                [[position_id]],
                device=input_ids.device
            )

            logits = self(
                next_token,
                position_ids=position_ids
            )

            if do_sample:
                next_token = _sample_next_token(logits[:, -1, :], temperature, top_p, top_k)
            else:
                next_token = logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True
                )

            
            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=1
            )

            # Then check whether generation should stop.
            if stop_token_ids is not None:
                if any(
                    torch.all(next_token == token_id)
                    for token_id in stop_token_ids
                ):
                    break

        return generated_ids

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, do_sample=False, temperature=1.0, top_p=None, top_k=None, stop_token_ids=None):
        if stop_token_ids is None:
            stop_token_ids = self.DEFAULT_STOP_TOKEN_IDS

        if self.use_kvcache:
            return self._generate_with_kv_cache(
                input_ids,
                max_new_tokens,
                do_sample,
                temperature,
                top_p,
                top_k,
                stop_token_ids=stop_token_ids,
            )

        return self._generate_without_kv_cache(
            input_ids,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            top_k,
            stop_token_ids=stop_token_ids,
        )
