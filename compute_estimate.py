"""
compute_estimate.py — forward-pass compute estimate for the three ablation models.

Counts multiply-accumulate operations (one multiply plus one add — the standard
unit for model compute, written MAC below) per single-image forward pass.

Per transformer block with N tokens, model width D, and feed-forward multiplier R:

  Linear terms (scale linearly with N):
    QKV projection        3 * N * D^2
    attention output proj     N * D^2
    feed-forward network  2 * R * N * D^2
    -----------------------------------
    subtotal              N * D^2 * (4 + 2R)

  Attention terms (scale with N^2 — the token-vs-token comparison):
    scores  Q . K^T           N^2 * D
    weighted sum of values    N^2 * D
    -----------------------------------
    subtotal              2 * N^2 * D

  block total = N * D^2 * (4 + 2R)  +  2 * N^2 * D

The linear term dominates at the small token counts here (tens of tokens);
the N^2 term only takes over at sequence lengths in the hundreds+. That is
why the compression saving is real but more modest than "attention is
quadratic" would suggest.
"""

D = 256          # model width, shared by all three models
HEADS = 8        # not needed for the count (heads partition D, total work unchanged)

def block_macs(n_tokens: float, d: int, mlp_ratio: float) -> tuple[float, float]:
    """Return (linear_macs, attention_macs) for one transformer block."""
    linear = n_tokens * d * d * (4 + 2 * mlp_ratio)
    attn   = 2 * (n_tokens ** 2) * d
    return linear, attn

def patch_embed_macs(n_patches: int, patch_px: int, in_ch: int, d: int) -> float:
    # Conv projection: each of n_patches outputs, D channels, from in_ch*patch^2 inputs
    return n_patches * d * (in_ch * patch_px * patch_px)

def router_macs(n_tokens: float, d: int, proj_dim: int) -> float:
    # Two projections D -> proj_dim (query/key side of the boundary scorer)
    return 2 * n_tokens * d * proj_dim

def merge_macs(n_tokens: float, d: int) -> float:
    # GroupMerge applies one D->D linear over the tokens
    return n_tokens * d * d


def summarize(name: str, blocks: list[tuple[str, float, float]],
              extras: list[tuple[str, float]]):
    """blocks: list of (label, n_tokens, mlp_ratio).  extras: list of (label, macs)."""
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    total_lin = total_attn = 0.0
    print(f"{'component':<24}{'tokens':>8}{'linear':>13}{'attn':>12}")
    print("-" * 60)
    for label, n, r in blocks:
        lin, att = block_macs(n, D, r)
        total_lin += lin
        total_attn += att
        print(f"{label:<24}{n:>8.0f}{lin/1e6:>11.1f}M{att/1e6:>10.1f}M")
    extra_total = 0.0
    for label, macs in extras:
        extra_total += macs
        print(f"{label:<24}{'':>8}{macs/1e6:>11.1f}M{'':>12}")
    grand = total_lin + total_attn + extra_total
    print("-" * 60)
    print(f"{'TOTAL':<24}{'':>8}{'':>12}{grand/1e6:>10.1f}M")
    print(f"  linear:     {total_lin/1e6:8.1f}M  ({100*total_lin/grand:4.1f}%)")
    print(f"  attention:  {total_attn/1e6:8.1f}M  ({100*total_attn/grand:4.1f}%)")
    print(f"  other:      {extra_total/1e6:8.1f}M  ({100*extra_total/grand:4.1f}%)")
    return grand


pe = patch_embed_macs(196, 16, 3, D)   # identical for all three
head = D * 100                          # classifier, negligible

# --- Plain Vision Transformer: 7 blocks at 196 tokens, feed-forward mult 4.0 ---
vit = summarize(
    "Plain Vision Transformer  (depth 7, ff x4.0)",
    [(f"block {i+1}", 196, 4.0) for i in range(7)],
    [("patch embed", pe), ("classifier", head)],
)

# --- Single-stage ASFNet: enc2 @196, main6 @65, feed-forward mult 3.0 ---
single = summarize(
    "Single-stage ASFNet  (enc2 + main6, ff x3.0)",
    [("encoder 1", 196, 3.0), ("encoder 2", 196, 3.0)]
    + [(f"main {i+1}", 65.33, 3.0) for i in range(6)],
    [("patch embed", pe),
     ("router (stage 1)", router_macs(196, D, 64)),
     ("merge (stage 1)",  merge_macs(196, D)),
     ("classifier", head)],
)

# --- Two-stage ASFNet: enc1_2 @196, enc2_2 @59, main4 @37, feed-forward mult 3.0 ---
# Token counts 59 and 37 are the MEASURED mean group counts from the run.
two = summarize(
    "Two-stage ASFNet  (enc2 + enc2 + main4, ff x3.0)",
    [("encoder1 1", 196, 3.0), ("encoder1 2", 196, 3.0),
     ("encoder2 1", 59, 3.0),  ("encoder2 2", 59, 3.0)]
    + [(f"main {i+1}", 37, 3.0) for i in range(4)],
    [("patch embed", pe),
     ("router (stage 1)", router_macs(196, D, 64)),
     ("merge (stage 1)",  merge_macs(196, D)),
     ("router (stage 2)", router_macs(59, D, 64)),
     ("merge (stage 2)",  merge_macs(59, D)),
     ("classifier", head)],
)

print(f"\n\n{'='*60}\nRELATIVE COST  (plain transformer = 1.00x)\n{'='*60}")
print(f"  Plain Vision Transformer :  {vit/1e6:8.1f}M   1.00x")
print(f"  Single-stage ASFNet      :  {single/1e6:8.1f}M   {single/vit:.2f}x")
print(f"  Two-stage ASFNet         :  {two/1e6:8.1f}M   {two/vit:.2f}x")
print(f"\n  Two-stage vs single-stage:  {two/single:.2f}x "
      f"({100*(1-two/single):.0f}% cheaper)")
