import torch
from .model import LlamaForCausalLM
from safetensors.torch import load_file


def load_llama_weights(model, hf_weight_path):
    hf_weights = load_file(
        hf_weight_path
    )

    def copy_weight(dst, src, name):
        if dst.shape != src.shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"model={tuple(dst.shape)}, "
                f"checkpoint={tuple(src.shape)}"
            )

        with torch.no_grad():
            dst.copy_(src.to(dtype=dst.dtype, device=dst.device))

    # --------------------------------------------------
    # Embedding
    # --------------------------------------------------
    copy_weight(
        model.model.embed_tokens.weight,
        hf_weights["model.embed_tokens.weight"],
        "embed_tokens.weight",
    )

    # --------------------------------------------------
    # Transformer layers
    # --------------------------------------------------
    for i, layer in enumerate(model.model.layers):

        prefix = f"model.layers.{i}"

        copy_weight(
            layer.input_layernorm.gamma,
            hf_weights[f"{prefix}.input_layernorm.weight"],
            f"{prefix}.input_layernorm.weight",
        )

        copy_weight(
            layer.self_attn.q_proj.weight,
            hf_weights[f"{prefix}.self_attn.q_proj.weight"],
            f"{prefix}.self_attn.q_proj.weight",
        )

        copy_weight(
            layer.self_attn.k_proj.weight,
            hf_weights[f"{prefix}.self_attn.k_proj.weight"],
            f"{prefix}.self_attn.k_proj.weight",
        )

        copy_weight(
            layer.self_attn.v_proj.weight,
            hf_weights[f"{prefix}.self_attn.v_proj.weight"],
            f"{prefix}.self_attn.v_proj.weight",
        )

        copy_weight(
            layer.self_attn.o_proj.weight,
            hf_weights[f"{prefix}.self_attn.o_proj.weight"],
            f"{prefix}.self_attn.o_proj.weight",
        )

        copy_weight(
            layer.post_attention_layernorm.gamma,
            hf_weights[f"{prefix}.post_attention_layernorm.weight"],
            f"{prefix}.post_attention_layernorm.weight",
        )

        copy_weight(
            layer.mlp.gate_proj.weight,
            hf_weights[f"{prefix}.mlp.gate_proj.weight"],
            f"{prefix}.mlp.gate_proj.weight",
        )

        copy_weight(
            layer.mlp.up_proj.weight,
            hf_weights[f"{prefix}.mlp.up_proj.weight"],
            f"{prefix}.mlp.up_proj.weight",
        )

        copy_weight(
            layer.mlp.down_proj.weight,
            hf_weights[f"{prefix}.mlp.down_proj.weight"],
            f"{prefix}.mlp.down_proj.weight",
        )

    # --------------------------------------------------
    # Final RMSNorm
    # --------------------------------------------------
    copy_weight(
        model.model.norm.gamma,
        hf_weights["model.norm.weight"],
        "model.norm.weight",
    )

    print("All Llama weights loaded successfully.")
