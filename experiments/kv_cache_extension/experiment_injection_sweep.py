"""
Injection Coefficient Sweep
============================
Test whether 37.7% P(Volt) can be improved by scaling the injection
coefficient. Also test: is the ceiling set by the 1D projection, or
by the magnitude?
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig
from chuk_lazarus.models_v2.families.gemma.model import clip_residual


def load_model():
    config = UnifiedPipelineConfig()
    pipeline = UnifiedPipeline.from_pretrained("google/gemma-3-4b-it", pipeline_config=config)
    return pipeline.model, pipeline.tokenizer


def forward_with_injection(model, query_ids, inject_vec, inject_layer=30):
    """Forward pass with residual injection at inject_layer (last position only)."""
    backbone = model.model
    config = backbone.config

    h = backbone.embed_tokens(query_ids)
    h = h * mx.array(config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    for i in range(len(backbone.layers)):
        layer = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer(h, mask=mask, cache=None)
        h = output.hidden_states

        if i == inject_layer:
            # Inject at last position
            h_last = h[:, -1:, :] + inject_vec.reshape(1, 1, -1).astype(h.dtype)
            h = mx.concatenate([h[:, :-1, :], h_last], axis=1)

    h = backbone.norm(h)
    if model.tie_word_embeddings:
        logits = backbone.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)

    probs = mx.softmax(logits[0, -1, :])
    mx.eval(probs)
    return probs


def get_residual_at_layer(model, prompt, layer):
    """Get residual at specified layer, last position."""
    tokenizer = None
    backbone = model.model
    config = backbone.config

    # Need tokenizer - get from model's vocab
    from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig
    cfg = UnifiedPipelineConfig()
    pipeline = UnifiedPipeline.from_pretrained("google/gemma-3-4b-it", pipeline_config=cfg)
    tokenizer = pipeline.tokenizer

    input_ids = mx.array([tokenizer.encode(prompt, add_special_tokens=True)])

    h = backbone.embed_tokens(input_ids)
    h = h * mx.array(config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    for i in range(layer + 1):
        layer_i = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer_i(h, mask=mask, cache=None)
        h = output.hidden_states

    mx.eval(h)
    return h[0, -1, :]  # (2560,)


def main():
    model, tokenizer = load_model()

    document = "Zarkov Industries, a cutting-edge robotics company, was founded in 2019 in the city of Voltara by engineer Dimitri Zarkov."
    full_context = f"<start_of_turn>user\n{document}\n\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"
    bare_query = "<start_of_turn>user\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"

    bare_ids = mx.array([tokenizer.encode(bare_query, add_special_tokens=True)])
    volt_id = 194328

    # Get answer token embedding
    backbone = model.model
    embed = backbone.embed_tokens.weight[volt_id]  # (2560,)
    embed_norm = float(mx.sqrt(mx.sum(embed ** 2)))
    embed_unit = embed / embed_norm
    mx.eval(embed, embed_unit)
    print(f"Volt embedding norm: {embed_norm:.2f}")

    # Get full-context residual at L30 for reference
    full_ids = mx.array([tokenizer.encode(full_context, add_special_tokens=True)])

    h_full = backbone.embed_tokens(full_ids)
    h_full = h_full * mx.array(backbone.config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h_full.dtype)
    global_mask = backbone._create_attention_mask(h_full, None)
    sliding_mask = (
        backbone._create_attention_mask(h_full, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )
    for i in range(31):
        layer = backbone.layers[i]
        is_global = backbone.config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer(h_full, mask=mask, cache=None)
        h_full = output.hidden_states
    donor_L30 = h_full[0, -1, :]
    mx.eval(donor_L30)

    donor_norm = float(mx.sqrt(mx.sum(donor_L30 ** 2)))
    coeff = float(mx.sum(donor_L30 * embed) / (mx.sum(embed * embed) + 1e-8))
    proj_norm = abs(coeff) * embed_norm
    print(f"Donor L30 residual norm: {donor_norm:.2f}")
    print(f"Projection coefficient: {coeff:.4f}")
    print(f"Projection norm: {proj_norm:.2f}")
    print(f"Projection / residual norm: {proj_norm/donor_norm:.4f}")

    # Also get bare query residual at L30
    h_bare = backbone.embed_tokens(bare_ids)
    h_bare = h_bare * mx.array(backbone.config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h_bare.dtype)
    bare_mask = backbone._create_attention_mask(h_bare, None)
    bare_slide = (
        backbone._create_attention_mask(h_bare, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )
    for i in range(31):
        layer = backbone.layers[i]
        is_global = backbone.config.is_global_layer(i)
        mask = bare_mask if is_global else bare_slide
        output = layer(h_bare, mask=mask, cache=None)
        h_bare = output.hidden_states
    bare_L30 = h_bare[0, -1, :]
    mx.eval(bare_L30)
    bare_norm = float(mx.sqrt(mx.sum(bare_L30 ** 2)))
    print(f"Bare query L30 residual norm: {bare_norm:.2f}")

    # ================================================================
    # SWEEP 1: Scale the coefficient
    # ================================================================
    print("\n" + "=" * 60)
    print("SWEEP 1: Injection magnitude at L30")
    print("=" * 60)

    scales = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    print(f"\n{'Scale':>8} {'Inject norm':>12} {'P(Volt)':>10} {'Top-1':>15}")
    print("-" * 50)

    for scale in scales:
        inject_vec = scale * coeff * embed
        inject_norm = float(mx.sqrt(mx.sum(inject_vec ** 2)))
        probs = forward_with_injection(model, bare_ids, inject_vec, inject_layer=30)
        p_volt = float(probs[volt_id])
        top_id = int(mx.argmax(probs))
        top_token = tokenizer.decode([top_id])
        top_prob = float(probs[top_id])
        print(f"  {scale:>6.1f}x {inject_norm:>10.1f} {p_volt:>10.4f} {top_token:>10} ({top_prob:.4f})")

    # ================================================================
    # SWEEP 2: Full residual vs 1D projection at L30
    # ================================================================
    print("\n" + "=" * 60)
    print("SWEEP 2: Full residual vs 1D at L30")
    print("=" * 60)

    # Full residual (donor - bare)
    diff = donor_L30 - bare_L30
    diff_norm = float(mx.sqrt(mx.sum(diff ** 2)))
    print(f"  Full diff norm: {diff_norm:.2f}")

    probs_full = forward_with_injection(model, bare_ids, diff, inject_layer=30)
    p_volt_full = float(probs_full[volt_id])
    print(f"  Full diff injection: P(Volt) = {p_volt_full:.6f}")

    # Just the 1D projection
    proj_1d = coeff * embed
    probs_1d = forward_with_injection(model, bare_ids, proj_1d, inject_layer=30)
    p_volt_1d = float(probs_1d[volt_id])
    print(f"  1D projection: P(Volt) = {p_volt_1d:.6f}")

    # Residual MINUS the 1D projection (orthogonal complement)
    ortho = diff - (float(mx.sum(diff * embed_unit)) * embed_unit)
    ortho_norm = float(mx.sqrt(mx.sum(ortho ** 2)))
    print(f"  Orthogonal complement norm: {ortho_norm:.2f}")

    probs_ortho = forward_with_injection(model, bare_ids, ortho, inject_layer=30)
    p_volt_ortho = float(probs_ortho[volt_id])
    print(f"  Orthogonal injection: P(Volt) = {p_volt_ortho:.6f}")

    # ================================================================
    # SWEEP 3: Inject raw donor L30 residual (full replacement)
    # ================================================================
    print("\n" + "=" * 60)
    print("SWEEP 3: Donor residual injection scales")
    print("=" * 60)

    for alpha in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
        inject_vec = alpha * donor_L30
        probs = forward_with_injection(model, bare_ids, inject_vec, inject_layer=30)
        p_volt = float(probs[volt_id])
        top_id = int(mx.argmax(probs))
        top_tok = tokenizer.decode([top_id])
        print(f"  alpha={alpha:.2f}: P(Volt)={p_volt:.4f}, top={top_tok} ({float(probs[top_id]):.4f})")


if __name__ == "__main__":
    main()
