import torch
import torch.nn as nn


class LlamaGQA(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, rope):
        super().__init__()

        assert d_model % n_q_heads  == 0, f"d_model ({d_model}) is not divisible by n_heads ({n_q_heads})"
        assert n_q_heads % n_kv_heads == 0, f"n_q_heads ({n_q_heads}) not divisible by n_kv_heads ({n_kv_heads})"

        self.d_model = d_model
        self.n_q_head = n_q_heads
        self.n_kv_head = n_kv_heads
        self.n_groups = n_q_heads // n_kv_heads

        self.head_dim = d_model // n_q_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim*n_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim*n_kv_heads, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = rope

    def forward(self, X, position_ids=None, mask=True, kvcache=None):
        # X: [B, N, d_model]
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        B, N, _ = X.shape

        q_head = Q.reshape(
            B, N, self.n_q_head, self.head_dim
        ).transpose(1, 2)

        k_head = K.reshape(
            B, N, self.n_kv_head, self.head_dim
        ).transpose(1, 2)

        v_head = V.reshape(
            B, N, self.n_kv_head, self.head_dim
        ).transpose(1, 2)

        if self.rope:
            # RoPE is applied ONLY to Q and K
            q_head = self.rope(q_head, position_ids)
            k_head = self.rope(k_head, position_ids)

        # ---------------------------------------------------------
        # KV cache
        # ---------------------------------------------------------

        if kvcache is not None:
            # Remember whether cache already contains previous tokens.
            is_prefill = kvcache.is_empty()

            # Append current K/V to cache.
            k_head, v_head = kvcache.update(
                k_head, v_head
            )
        else:
            # No cache is being used.
            is_prefill = True

        q_len = q_head.shape[2]
        kv_len = k_head.shape[2]

        # [B, n_q_heads, q_len, head_dim]
        # ->
        # [B, n_kv_heads, n_groups, q_len, head_dim]
        q_head = q_head.reshape(
            B,
            self.n_kv_head,
            self.n_groups,
            q_len,
            self.head_dim
        )

        scores = torch.einsum(
            "bhgqd,bhkd->bhgqk",
            q_head,
            k_head
        ) / (self.head_dim ** 0.5)

        # ---------------------------------------------------------
        # Causal mask
        # ---------------------------------------------------------

        # Mask when:
        # 1. no KV cache is being used, OR
        # 2. this is the first pass (prefill) with a KV cache.
        #
        # During cached decoding, q_len = 1 and the query can
        # legitimately attend to every key already in the cache,
        # so no mask is required.
        if mask and is_prefill:
            causal_mask = torch.triu(
                torch.ones(
                    q_len,
                    kv_len,
                    device=X.device,
                    dtype=torch.bool
                ),
                diagonal=1
            )

            scores = scores.masked_fill(
                causal_mask,
                float("-inf")
            )

        attn = torch.softmax(scores, dim=-1)

        output = torch.einsum(
            "bhgqk,bhkd->bhgqd",
            attn,
            v_head
        )

        # [B, n_kv_heads, n_groups, q_len, head_dim]
        # ->
        # [B, n_q_heads, q_len, head_dim]
        output = output.reshape(
            B,
            self.n_q_head,
            q_len,
            self.head_dim
        )

        output = output.permute(0, 2, 1, 3)

        # [B, q_len, d_model]
        output = output.reshape(
            B,
            q_len,
            self.d_model
        )

        return self.o_proj(output)
