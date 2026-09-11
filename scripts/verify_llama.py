import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from llama3_2_1b_scratch.model import LlamaForCausalLM
from llama3_2_1b_scratch.load_weights import load_llama_weights


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "models/Llama-3.2-1B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# HF and scratch can use different kernels / operation ordering.
TOLERANCE = 0.25


# =========================================================
# HELPERS
# =========================================================

def max_diff(a, b):
    """
    Maximum absolute difference between two tensors.
    Convert to float32 first so comparison is stable.
    """
    return (a.float() - b.float()).abs().max().item()


def print_diff(name, diff):
    print(f"{name:<25}: {diff}")


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("LLAMA 3.2 1B SCRATCH IMPLEMENTATION VERIFICATION")
    print("=" * 70)

    # =====================================================
    # 1. LOAD HF MODEL
    # =====================================================

    print("\nLoading Hugging Face model...")

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
    ).to(DEVICE)

    hf_model.eval()

    # =====================================================
    # 2. LOAD SCRATCH MODEL
    # =====================================================

    print("Loading scratch model...")

    scratch_model = LlamaForCausalLM().to(
        device=DEVICE,
        dtype=DTYPE,
    )

    scratch_model.eval()

    # =====================================================
    # 3. LOAD WEIGHTS
    # =====================================================

    print("Loading scratch weights...")

    load_llama_weights(
        scratch_model,
        f"{MODEL_PATH}/model.safetensors",
    )

    print("Weights loaded.")

    # =====================================================
    # 4. TOKENIZER + INPUT
    # =====================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
    )

    text = "The capital of France is"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = inputs["input_ids"].to(DEVICE)

    batch_size, seq_len = input_ids.shape

    print("\n" + "=" * 70)
    print("INPUT")
    print("=" * 70)

    print("Text :", repr(text))
    print("Shape:", input_ids.shape)
    print("IDs  :", input_ids)

    # =====================================================
    # GET LAYER 0
    # =====================================================

    hf_layer = hf_model.model.layers[0]
    scratch_layer = scratch_model.model.layers[0]

    hf_attn = hf_layer.self_attn
    scratch_attn = scratch_layer.self_attn

    # =====================================================
    # RESULTS
    # =====================================================

    results = {}

    # =====================================================
    # 5. EMBEDDING
    # =====================================================

    print("\n" + "-" * 70)
    print("1. EMBEDDING")
    print("-" * 70)

    with torch.no_grad():

        hf_embed = hf_model.model.embed_tokens(
            input_ids
        )

        scratch_embed = scratch_model.model.embed_tokens(
            input_ids
        )

    embedding_diff = max_diff(
        hf_embed,
        scratch_embed,
    )

    print_diff(
        "Embedding max diff",
        embedding_diff,
    )

    results["Embedding"] = embedding_diff

    # =====================================================
    # 6. FIRST RMSNORM
    # =====================================================

    print("\n" + "-" * 70)
    print("2. FIRST RMSNORM")
    print("-" * 70)

    with torch.no_grad():

        hf_norm = hf_layer.input_layernorm(
            hf_embed
        )

        scratch_norm = scratch_layer.input_layernorm(
            scratch_embed
        )

    rmsnorm_diff = max_diff(
        hf_norm,
        scratch_norm,
    )

    print_diff(
        "RMSNorm max diff",
        rmsnorm_diff,
    )

    results["First RMSNorm"] = rmsnorm_diff

    # =====================================================
    # 7. Q / K / V
    # =====================================================

    print("\n" + "-" * 70)
    print("3. Q / K / V BEFORE RoPE")
    print("-" * 70)

    with torch.no_grad():

        hf_q = hf_attn.q_proj(
            hf_norm
        )

        hf_k = hf_attn.k_proj(
            hf_norm
        )

        hf_v = hf_attn.v_proj(
            hf_norm
        )

        scratch_q = scratch_attn.q_proj(
            scratch_norm
        )

        scratch_k = scratch_attn.k_proj(
            scratch_norm
        )

        scratch_v = scratch_attn.v_proj(
            scratch_norm
        )

    q_diff = max_diff(
        hf_q,
        scratch_q,
    )

    k_diff = max_diff(
        hf_k,
        scratch_k,
    )

    v_diff = max_diff(
        hf_v,
        scratch_v,
    )

    print_diff("Q max diff", q_diff)
    print_diff("K max diff", k_diff)
    print_diff("V max diff", v_diff)

    results["Q"] = q_diff
    results["K"] = k_diff
    results["V"] = v_diff

    # =====================================================
    # 8. HEAD RESHAPE
    # =====================================================

    print("\n" + "-" * 70)
    print("4. HEAD RESHAPE")
    print("-" * 70)

    # IMPORTANT:
    # These are the names used by YOUR scratch implementation.

    n_q_heads = scratch_attn.n_q_head
    n_kv_heads = scratch_attn.n_kv_head
    head_dim = scratch_attn.head_dim

    print("Q heads :", n_q_heads)
    print("KV heads:", n_kv_heads)
    print("Head dim:", head_dim)

    with torch.no_grad():

        # -------------------------
        # HF
        # -------------------------

        hf_q_head = hf_q.reshape(
            batch_size,
            seq_len,
            n_q_heads,
            head_dim,
        ).transpose(1, 2)

        hf_k_head = hf_k.reshape(
            batch_size,
            seq_len,
            n_kv_heads,
            head_dim,
        ).transpose(1, 2)

        hf_v_head = hf_v.reshape(
            batch_size,
            seq_len,
            n_kv_heads,
            head_dim,
        ).transpose(1, 2)

        # -------------------------
        # Scratch
        # -------------------------

        scratch_q_head = scratch_q.reshape(
            batch_size,
            seq_len,
            n_q_heads,
            head_dim,
        ).transpose(1, 2)

        scratch_k_head = scratch_k.reshape(
            batch_size,
            seq_len,
            n_kv_heads,
            head_dim,
        ).transpose(1, 2)

        scratch_v_head = scratch_v.reshape(
            batch_size,
            seq_len,
            n_kv_heads,
            head_dim,
        ).transpose(1, 2)

    q_head_diff = max_diff(
        hf_q_head,
        scratch_q_head,
    )

    k_head_diff = max_diff(
        hf_k_head,
        scratch_k_head,
    )

    v_head_diff = max_diff(
        hf_v_head,
        scratch_v_head,
    )

    print_diff("Q head max diff", q_head_diff)
    print_diff("K head max diff", k_head_diff)
    print_diff("V head max diff", v_head_diff)

    results["Q Head Reshape"] = q_head_diff
    results["K Head Reshape"] = k_head_diff
    results["V Head Reshape"] = v_head_diff

    # =====================================================
    # 9. RoPE
    # =====================================================

    print("\n" + "-" * 70)
    print("5. RoPE")
    print("-" * 70)

    position_ids = torch.arange(
        seq_len,
        device=DEVICE,
    ).unsqueeze(0)

    # -----------------------------------------------------
    # Scratch RoPE
    # -----------------------------------------------------

    with torch.no_grad():

        scratch_q_rope = scratch_attn.rope(
            scratch_q_head
        )

        scratch_k_rope = scratch_attn.rope(
            scratch_k_head
        )

    # -----------------------------------------------------
    # HF RoPE
    #
    # IMPORTANT:
    # Your installed HF version uses:
    #
    # hf_model.model.rotary_emb(...)
    #
    # followed by apply_rotary_pos_emb(...)
    # -----------------------------------------------------

    with torch.no_grad():

        hf_cos, hf_sin = hf_model.model.rotary_emb(
            hf_q_head,
            position_ids,
        )

        hf_q_rope, hf_k_rope = apply_rotary_pos_emb(
            hf_q_head,
            hf_k_head,
            hf_cos,
            hf_sin,
        )

    q_rope_diff = max_diff(
        hf_q_rope,
        scratch_q_rope,
    )

    k_rope_diff = max_diff(
        hf_k_rope,
        scratch_k_rope,
    )

    print("HF cos shape    :", hf_cos.shape)
    print("HF sin shape    :", hf_sin.shape)

    print_diff(
        "RoPE Q max diff",
        q_rope_diff,
    )

    print_diff(
        "RoPE K max diff",
        k_rope_diff,
    )

    results["RoPE Q"] = q_rope_diff
    results["RoPE K"] = k_rope_diff

    # =====================================================
    # 10. GQA KV EXPANSION
    # =====================================================

    print("\n" + "-" * 70)
    print("6. GQA KV EXPANSION")
    print("-" * 70)

    repeat_factor = n_q_heads // n_kv_heads

    print("Repeat factor:", repeat_factor)

    with torch.no_grad():

        # HF
        hf_k_expanded = hf_k_rope.repeat_interleave(
            repeat_factor,
            dim=1,
        )

        hf_v_expanded = hf_v_head.repeat_interleave(
            repeat_factor,
            dim=1,
        )

        # Scratch
        scratch_k_expanded = scratch_k_rope.repeat_interleave(
            repeat_factor,
            dim=1,
        )

        scratch_v_expanded = scratch_v_head.repeat_interleave(
            repeat_factor,
            dim=1,
        )

    gqa_k_diff = max_diff(
        hf_k_expanded,
        scratch_k_expanded,
    )

    gqa_v_diff = max_diff(
        hf_v_expanded,
        scratch_v_expanded,
    )

    print_diff(
        "Expanded K max diff",
        gqa_k_diff,
    )

    print_diff(
        "Expanded V max diff",
        gqa_v_diff,
    )

    results["GQA K"] = gqa_k_diff
    results["GQA V"] = gqa_v_diff

    # =====================================================
    # 11. ATTENTION SCORES
    # =====================================================

    print("\n" + "-" * 70)
    print("7. ATTENTION SCORES")
    print("-" * 70)

    scale = 1.0 / (head_dim ** 0.5)

    with torch.no_grad():

        hf_scores = (
            hf_q_rope
            @ hf_k_expanded.transpose(-2, -1)
        ) * scale

        scratch_scores = (
            scratch_q_rope
            @ scratch_k_expanded.transpose(-2, -1)
        ) * scale

    score_diff = max_diff(
        hf_scores,
        scratch_scores,
    )

    print_diff(
        "Attention scores max diff",
        score_diff,
    )

    results["Attention Scores"] = score_diff

    # =====================================================
    # 12. CAUSAL MASK + SOFTMAX
    # =====================================================

    print("\n" + "-" * 70)
    print("8. ATTENTION PROBABILITIES")
    print("-" * 70)

    causal_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            device=DEVICE,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    with torch.no_grad():

        hf_scores_masked = hf_scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        scratch_scores_masked = scratch_scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        hf_probs = torch.softmax(
            hf_scores_masked,
            dim=-1,
        )

        scratch_probs = torch.softmax(
            scratch_scores_masked,
            dim=-1,
        )

    prob_diff = max_diff(
        hf_probs,
        scratch_probs,
    )

    print_diff(
        "Attention probs max diff",
        prob_diff,
    )

    results["Attention Probs"] = prob_diff

    # =====================================================
    # 13. ATTENTION × V
    # =====================================================

    print("\n" + "-" * 70)
    print("9. ATTENTION CONTEXT")
    print("-" * 70)

    with torch.no_grad():

        hf_context = hf_probs @ hf_v_expanded

        scratch_context = (
            scratch_probs @ scratch_v_expanded
        )

    context_diff = max_diff(
        hf_context,
        scratch_context,
    )

    print_diff(
        "Context max diff",
        context_diff,
    )

    results["Attention Context"] = context_diff

    # =====================================================
    # 14. CONCATENATE HEADS
    # =====================================================

    print("\n" + "-" * 70)
    print("10. CONTEXT RESHAPE")
    print("-" * 70)

    hf_context_flat = (
        hf_context
        .transpose(1, 2)
        .reshape(
            batch_size,
            seq_len,
            n_q_heads * head_dim,
        )
    )

    scratch_context_flat = (
        scratch_context
        .transpose(1, 2)
        .reshape(
            batch_size,
            seq_len,
            n_q_heads * head_dim,
        )
    )

    context_flat_diff = max_diff(
        hf_context_flat,
        scratch_context_flat,
    )

    print_diff(
        "Context reshape max diff",
        context_flat_diff,
    )

    results["Context Reshape"] = context_flat_diff

    # =====================================================
    # 15. O PROJECTION
    # =====================================================

    print("\n" + "-" * 70)
    print("11. O PROJECTION")
    print("-" * 70)

    with torch.no_grad():

        hf_o = hf_attn.o_proj(
            hf_context_flat
        )

        scratch_o = scratch_attn.o_proj(
            scratch_context_flat
        )

    o_diff = max_diff(
        hf_o,
        scratch_o,
    )

    print_diff(
        "O projection max diff",
        o_diff,
    )

    results["O Projection"] = o_diff

    # =====================================================
    # 16. ACTUAL HF ATTENTION VS SCRATCH ATTENTION
    # =====================================================

    print("\n" + "-" * 70)
    print("12. ACTUAL ATTENTION OUTPUT")
    print("-" * 70)

    with torch.no_grad():

        # HF attention
        hf_attn_output = hf_attn(
            hf_norm,
            position_embeddings=(hf_cos, hf_sin),
            attention_mask=None,
            past_key_values=None,
            cache_position=torch.arange(
                position_ids.shape[-1],
                device=DEVICE,
            ),
        )[0]

        # Scratch attention
        scratch_attn_output = scratch_attn(
            scratch_norm,
            position_ids=position_ids,
        )

    actual_attention_diff = max_diff(
        hf_attn_output,
        scratch_attn_output,
    )

    print_diff(
        "Actual attention max diff",
        actual_attention_diff,
    )

    results["Actual Attention"] = actual_attention_diff

    # =====================================================
    # 17. RESIDUAL
    # =====================================================

    print("\n" + "-" * 70)
    print("13. POST-ATTENTION RESIDUAL")
    print("-" * 70)

    # Correct residual:
    #
    # residual = original hidden state + attention output
    #
    # NOT:
    #
    # normalized hidden state + attention output

    with torch.no_grad():

        hf_residual = (
            hf_embed +
            hf_attn_output
        )

        scratch_residual = (
            scratch_embed +
            scratch_attn_output
        )

    residual_diff = max_diff(
        hf_residual,
        scratch_residual,
    )

    print_diff(
        "Residual max diff",
        residual_diff,
    )

    results["Residual"] = residual_diff

    # =====================================================
    # 18. SAME-INPUT RMSNORM
    # =====================================================

    print("\n" + "-" * 70)
    print("14. SAME-INPUT POST-ATTENTION RMSNORM")
    print("-" * 70)

    # Feed the EXACT SAME tensor into both implementations.
    #
    # This isolates the RMSNorm implementation from any
    # preceding attention numerical differences.

    with torch.no_grad():

        hf_post_norm = hf_layer.post_attention_layernorm(
            hf_residual
        )

        scratch_post_norm = (
            scratch_layer.post_attention_layernorm(
                hf_residual
            )
        )

    same_input_norm_diff = max_diff(
        hf_post_norm,
        scratch_post_norm,
    )

    print_diff(
        "Same-input RMSNorm max diff",
        same_input_norm_diff,
    )

    results["Same-input RMSNorm"] = same_input_norm_diff

    # =====================================================
    # 19. SAME-INPUT MLP
    # =====================================================

    print("\n" + "-" * 70)
    print("15. SAME-INPUT MLP")
    print("-" * 70)

    hf_mlp = hf_layer.mlp
    scratch_mlp = scratch_layer.mlp

    # IMPORTANT:
    #
    # Both implementations receive the EXACT SAME tensor.
    #
    # Therefore this directly tests:
    #
    # gate_proj
    # up_proj
    # SiLU
    # gate * up
    # down_proj
    #
    # without residual/attention numerical differences.

    mlp_input = hf_post_norm

    with torch.no_grad():

        # -------------------------
        # HF
        # -------------------------

        hf_gate = hf_mlp.gate_proj(
            mlp_input
        )

        hf_up = hf_mlp.up_proj(
            mlp_input
        )

        hf_silu = torch.nn.functional.silu(
            hf_gate
        )

        hf_gate_up = (
            hf_silu *
            hf_up
        )

        hf_down = hf_mlp.down_proj(
            hf_gate_up
        )

        # -------------------------
        # Scratch
        # -------------------------

        scratch_gate = scratch_mlp.gate_proj(
            mlp_input
        )

        scratch_up = scratch_mlp.up_proj(
            mlp_input
        )

        scratch_silu = torch.nn.functional.silu(
            scratch_gate
        )

        scratch_gate_up = (
            scratch_silu *
            scratch_up
        )

        scratch_down = scratch_mlp.down_proj(
            scratch_gate_up
        )

    same_gate_diff = max_diff(
        hf_gate,
        scratch_gate,
    )

    same_up_diff = max_diff(
        hf_up,
        scratch_up,
    )

    same_silu_diff = max_diff(
        hf_silu,
        scratch_silu,
    )

    same_gate_up_diff = max_diff(
        hf_gate_up,
        scratch_gate_up,
    )

    same_down_diff = max_diff(
        hf_down,
        scratch_down,
    )

    print_diff(
        "Gate projection max diff",
        same_gate_diff,
    )

    print_diff(
        "Up projection max diff",
        same_up_diff,
    )

    print_diff(
        "SiLU max diff",
        same_silu_diff,
    )

    print_diff(
        "Gate × Up max diff",
        same_gate_up_diff,
    )

    print_diff(
        "Down projection max diff",
        same_down_diff,
    )

    results["MLP Gate"] = same_gate_diff
    results["MLP Up"] = same_up_diff
    results["MLP SiLU"] = same_silu_diff
    results["MLP Gate × Up"] = same_gate_up_diff
    results["MLP Down"] = same_down_diff

    # =====================================================
    # 20. FINAL LOGITS
    # =====================================================

    print("\n" + "-" * 70)
    print("16. FINAL LOGITS")
    print("-" * 70)

    with torch.no_grad():

        hf_logits = hf_model(
            input_ids
        ).logits[:, -1, :]

        scratch_logits = scratch_model(
            input_ids
        )[:, -1, :]

    logits_diff = max_diff(
        hf_logits,
        scratch_logits,
    )

    print_diff(
        "Final logits max diff",
        logits_diff,
    )

    results["Final Logits"] = logits_diff

    # =====================================================
    # 21. TOP-1
    # =====================================================

    print("\n" + "-" * 70)
    print("17. TOP-1 PREDICTION")
    print("-" * 70)

    hf_top1 = hf_logits.argmax(
        dim=-1
    ).item()

    scratch_top1 = scratch_logits.argmax(
        dim=-1
    ).item()

    hf_token = tokenizer.decode(
        [hf_top1],
        clean_up_tokenization_spaces=False,
    )

    scratch_token = tokenizer.decode(
        [scratch_top1],
        clean_up_tokenization_spaces=False,
    )

    print(
        "HF     :",
        repr(hf_token),
        f"(ID {hf_top1})",
    )

    print(
        "Scratch:",
        repr(scratch_token),
        f"(ID {scratch_top1})",
    )

    top1_pass = (
        hf_top1 ==
        scratch_top1
    )

    results["Top-1"] = top1_pass

    # =====================================================
    # 22. TOP-5
    # =====================================================

    print("\n" + "-" * 70)
    print("18. TOP-5 PREDICTIONS")
    print("-" * 70)

    hf_top5_values, hf_top5_ids = torch.topk(
        hf_logits,
        5,
        dim=-1,
    )

    scratch_top5_values, scratch_top5_ids = torch.topk(
        scratch_logits,
        5,
        dim=-1,
    )

    print("\nHF:")

    for token_id, value in zip(
        hf_top5_ids[0],
        hf_top5_values[0],
    ):

        token_id = token_id.item()

        print(
            f"{token_id:6d}",
            repr(
                tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False,
                )
            ),
            f"{value.item():.4f}",
        )

    print("\nScratch:")

    for token_id, value in zip(
        scratch_top5_ids[0],
        scratch_top5_values[0],
    ):

        token_id = token_id.item()

        print(
            f"{token_id:6d}",
            repr(
                tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False,
                )
            ),
            f"{value.item():.4f}",
        )

    top5_pass = torch.equal(
        hf_top5_ids,
        scratch_top5_ids,
    )

    results["Top-5"] = top5_pass

    # =====================================================
    # 23. SUMMARY
    # =====================================================

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    numerical_pass = True
    categorical_pass = True

    for name, value in results.items():

        if isinstance(value, bool):

            status = (
                "PASS"
                if value
                else "FAIL"
            )

            categorical_pass &= value

            print(
                f"{name:<25}: {status}"
            )

        else:

            status = (
                "PASS"
                if value <= TOLERANCE
                else "FAIL"
            )

            numerical_pass &= (
                value <= TOLERANCE
            )

            print(
                f"{name:<25}: "
                f"{status} "
                f"({value:.6f})"
            )

    overall = (
        numerical_pass
        and categorical_pass
    )

    print("\n" + "=" * 70)
    print(
        f"OVERALL: "
        f"{'PASS' if overall else 'FAIL'}"
    )
    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()