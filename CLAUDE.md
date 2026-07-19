# NeocoreImagenet

Neocore (formerly ASFNet — old naming retired) experiments on ImageNet-100
(`clane9/imagenet-100`, auto-downloaded from HuggingFace on first run; DALI
dataloading via a one-time JPEG cache, or the RAM-resident blob cache).

- `model_neocore.py` / `train_neocore.py` / `evaluate_neocore.py` — the
  **current model**: recursive AR-admission autoencoder (see the 2026-07-17
  checkpoint below). Eval renders admission-order maps + reconstructions +
  per-round decodes.
- `dataset_ram.py` — RAM-resident dataset (`--data ram`): decoded uint8
  256×256 blobs built once via DALI, torch-only loader with GPU aug;
  `--data_device cuda` keeps the whole dataset in VRAM. Val protocol exactly
  equivalent to DALI val (numbers comparable).
- `train_asfnet*.py` / `model_asfnet*.py` — legacy classification variants
  (`_br` = border-retention, `2` = two-stage); `*_ae` — the MAE-style AE /
  ladder era (eval: `evaluate_asfnet_br.py --ae`).
- `train_linear_probe.py` — frozen-backbone linear/attentive probe of an AE
  checkpoint (wandb artifact); supports both Neocore and legacy backbones.
- wandb projects: `neocore` for Neocore AE runs, `asfnetAE` legacy AE,
  `asfnet` for probes.

## Research checkpoint (2026-07-15) — where the AE work stands

Full history in the wandb projects; the state of play:

1. **Baseline AE collapses to keep-everything.** 300-epoch run `cgyu2m9e`:
   final drop_frac 0.013, mean_kept 193.5/196, val rec 0.379 (on the ~1% it
   dropped). Mechanism: the edge-level ratio loss is *satisfied* at ~64%
   cuts (its value 1.0 IS the normalised minimum) in any spatial
   arrangement, and at target_group_size 3 chunks have no interiors, so
   border-retention keeps ~all tokens and reconstruction is trivial. There
   is no pressure toward compression in that objective.
2. **Variant A — hard bottleneck (`--keep_budget 0.25`) WORKS.** Run
   `wx8mlobe` (`AE_budget25`): top-49-by-evidence survivors enter the
   decoder, rest masked + counted as dropped. drop_frac pinned 0.75, val
   rec 0.101, legible reconstructions (user-confirmed). This is the
   working instrument going forward.
3. **Variant B — soft token keep-rate loss FAILED TWICE**, each time by
   finding a degenerate solution (both were exploits the user predicted):
   `s3vuij4s`: collapse all edge probs to 0 → zero-border guard keeps all →
   the loss's (F=1, G=0) corner is exactly 0 (fixed by thresholding soft
   p_keep for F). `0ie7me7e` (v2): park all edge probs in ~[0.16, 0.5) so
   accumulated soft evidence hits the 25% target while NO edge crosses the
   0.5 hard-cut threshold → zero cuts → guard keeps all, rec loss 0, stable
   fixed point (killed at epoch 15). **Lesson: any soft pressure + hard
   drop threshold + loss-only-on-dropped lets SGD park just under the
   threshold.** A third attempt would need retention itself redefined on
   accumulated evidence (keep ⇔ p_keep > 0.5) — an architecture change,
   user's call, not yet made.
4. **Linear probe of A** (`probe_AE_budget25`, project `asfnet`): top-1
   8.3% / top-5 23.5%. Known caveats: probe mean-pools ALL ~190
   router-kept tokens but only the top-49 got reconstruction gradient
   (pooling mismatch), and MAE-style features probe poorly in general.
   Candidate follow-ups: pool only top-49-evidence tokens, attentive
   probe, or full fine-tune (the honest ceiling).

Agreed next directions (not started): budget annealing for A (e.g. 0.9 →
0.25), rate-distortion learnable K, fine-tune script (small variant of
train_linear_probe.py with everything unfrozen + lower LR).

## Research checkpoint (2026-07-16) — probes, slots, and the two-stage AEs

All checkpoints + eval panels now live in local `runs/<run_name>/`
(wandb = metrics only). Loss conventions: "dropped-only" vs "all-position"
rec numbers are NOT comparable across conventions.

1. **Probe grid on frozen AE_budget25** (180 epochs each): attentive+ALL
   survivors **32.0%** top-1 / 59.3% top-5; attentive+top49 26.5%;
   mean+top49 12.1% (original mean+all 50ep: 8.3%). Lessons: the attentive
   (MAP) head is the dominant factor (+14pt); restricting to the 49
   reconstruction-graded tokens HURTS (-5.5pt) — the other ~140 survivors
   carry real class signal; the old 8.3% was a measurement artifact.
2. **Single-stage slot bottleneck (AE_xattn49) works**: 49 learned queries
   cross-attend over survivors, Perceiver-IO decode, loss on ALL positions.
   val rec 0.211 (original) / 0.227 (resume from ep174 after the 2026-07-15
   wandb storage outage ate the original's final weights; resume weights
   are in runs/AE_xattn49_resume/). Router stayed non-degenerate (~26% drop).
3. **Two-stage AE head-to-head (all 300 ep, enc 3+3, main 6) — hierarchy
   is currently an optimization problem, not a capacity one:**
   - pool (AE2): router2 group count THRASHED 1.7↔91 all run; rec 0.62
     all-position. Hypothesis: group count is a percolation quantity and
     tgs=3 sits near criticality; two stacked near-critical routers drive
     each other.
   - double retention (AE2BR): keep-all basin, only 12.5% dropped,
     rec 0.116 dropped-only = trivial. Same corner as the single-stage
     baseline collapse.
   - double retention + 49 slots: total collapse — routers thrashed, slots
     output a constant, rec pinned at 1.00 all-position from epoch 26.
   - double retention + budget 49 (rank by s1+s2): the ONLY stable
     two-stage — drop pinned 0.75, kept2=49, smooth descent — but final
     rec 0.218 dropped-only vs single-stage budget25's 0.101 at the same
     rate/convention. **Hard budgets stabilize two-stage training; the
     hierarchy still costs ~2x reconstruction.**
4. **Fine-patch ladder (AEL_4px_3stage) — granularity hypothesis supported.**
   4x4 patches (3136 tokens), 3 budgeted stages 784/196/49, 4.7M params,
   ~90 s/epoch (~$7.5/300ep). Trained STABLY (no thrash, no keep-all) and
   — first time for any variant — settled BELOW its final budget: stage 3
   keeps ~30 of 49 allowed (kept_s1 pinned 784), i.e. retention itself
   became the operative compression at fine granularity, where chunks have
   real interiors. Final val rec 0.361 while reconstructing 99.05% of the
   fine grid from ~30 tokens (~105x token compression; not numerically
   comparable to 16x16 runs — different patch size and mask rate). Rec was
   still falling at epoch 300 (undertrained). Weights/panels:
   runs/AEL_4px_3stage/.
5. **Ladder probe**: attentive pool over the final ~24 survivors = 26.3%
   top-1 — matches budget25's top-49 probe (26.5%) from HALF the tokens
   (budget25 all-190-survivors remains the absolute best at 32.0%).
6. **The enclosure diagnosis (the session's core insight).** Per-edge cuts
   need not CLOSE: a mesh of slits makes every token "border" without
   enclosing anything, so the architecture never actually asked for
   chunks. In 1D (H-Net) every cut IS a boundary; the 2D lift broke that
   identity, and all routing degeneracies trace to it. Instrumentation
   (visualize_ladder_stages.py, chunk maps + subgraph stats) showed the
   edge ladder's stage 3 cuts ZERO edges — its ~24 kept tokens are purely
   stage-2 islands.
7. **Two enclosure-aware routers tried** (--router_kind on the ladder):
   `field` (quantized scalar potential — every cut closes by topology)
   shattered stage 1 into singletons (97-99% cuts, closure vacuous) then
   went constant above, sliding into the ISLAND FUNNEL (~9 tokens by
   ep 125, later routers gradient-frozen). `component` (border = only
   component-separating cuts) collapsed to 6 island tokens by ep 25 —
   zero valid edges downstream = zero gradient, terminal. Root cause of
   both: budgets were CAPS with no floor; islands are free retention.
8. **Fix that holds: --budget_floor (exact-K per stage** — borders take
   priority, accumulated evidence fills the deficit). AEL_compfloor_4px:
   rates pinned exactly 784/196/49 and val rec 0.361 at epoch 50 —
   equal to the edge ladder's FINAL 300-epoch value at 1/6 the epochs.
   The routers still barely cut (retention is mostly evidence-fill +
   islands): enclosure-chunking remains UNACHIEVED; exact-K + evidence
   ranking is empirically the strongest AE mechanism so far.
9. **The session's law: any degree of freedom that lets the router
   modulate HOW MANY tokens survive gets used to escape the
   reconstruction task; only architecturally fixed quantities hold.**
   (Corners found to date: keep-all, threshold-parking x2, pool-thrash,
   slit-mesh border inflation, island under-supply x2, field shatter.)
10. Open next: why routers under-cut even with rates pinned (do cuts help
    rec at all under exact-K?); enclosure via region parametrization
    (differentiable superpixels) rather than cut fields; probe of
    compfloor when done; distillation/JEPA objective.

## Research checkpoint (2026-07-16 night) — ladder era closes; loop era opens

Final ladder results (4x4, 300 ep, rec on dropped @ 49-token rate unless
noted; all weights/panels in runs/<name>/):

1. **AEL_compfloor_4px: 0.270 — the project's best AE.** Component borders
   + exact-K floor. Stage-2 router grew to ~86 REAL separating cuts by the
   end (the floor un-froze it); stage 3 stayed islands+evidence-fill.
2. **AEL_ghost_4px (finishing overnight, ~0.30): the decisive negative.**
   Ghost grid fixed every mechanical grievance — dropped tokens persist as
   frozen ghosts, all 6160 edges routable at every stage, zero islands
   possible, resurrection allowed, ALL THREE routers demonstrably cutting
   (~4k edges each) — and the stages still produce the SAME-density cut
   fields. No coarsening across stages. Mechanics were never the blocker:
   a one-shot knowledge-free router cannot parse, no matter how healthy.
3. **AEL_pureborder_4px (killed ep ~55, batch-512/LR instability): the
   accidental positive.** Its stage-1 borders coarsened across TRAINING
   TIME (2789 -> 2081) while never coarsening across stages. Coarsening
   lives where knowledge accumulates (weights over epochs), not where it
   doesn't (depth within a pass).
4. **Synthesis (see POINTS_OF_INTEREST.md, local/gitignored — read it
   after this file): chunking is a POSTCONDITION of recognition.** The
   degeneracy zoo was SGD correctly reporting that stage-0 knowledge-free
   chunking has no solution. The evidence votes admission-control over
   parse-tree: every rec gain this week came from shedding region
   machinery (evidence rank + exact-K), every failure from demanding it.
5. **Next experiment (decided): recursive single-stage AR admission.**
   No ladder, no cuts, no borders. One weight-shared encoder applied
   recursively: each round scores tokens, admits K/R more (exact-K per
   round — the only regime that has ever held), marks admitted tokens
   with one learned "in-memory" embedding, re-encodes. R=1 == current
   budget model (built-in control). Reconstruction teaches; the attentive
   probe measures; ADMISSION-ORDER MAPS are the new instrument (color by
   round — watch where it looks once it knows what it has). Prediction to
   test: round-t admissions anti-correlate with what rounds <t already
   reconstruct.

Cloud lessons (2026-07-15/16): wandb GCS 403 outage (~1 h) and an HF 502
each killed runs at boot — artifact/dataset I/O now retries with backoff;
successful runs AWAIT PULL (`launch.py pull`, scp + account ssh key)
instead of self-destroying; known-bad machines list grew (m48680 GPU hang,
m140634 zombie boot).

## Research checkpoint (2026-07-17 overnight) — Neocore: the loop exists and it helps

Neocore built, validated, and R-swept in one night. Architecture
(model_neocore.py): one weight-shared 8-block core (d=256, h=8, mlp 3.0)
applied R rounds over the full 196-token 16×16 grid; each round a linear
score head ranks not-yet-admitted tokens, exactly K/R are admitted (hard
detached top-k — the only regime that has ever held), admitted tokens are
stamped once with a confidence-gated residual + learned marker; features
carry forward between rounds; MAE decoder (128×4) from the final K=49;
dropped-only norm-pix loss. NO auxiliary losses — K and R are
architectural constants (the law). 6.25M params. Nullity alarms logged
every step: `overlap_r1` (final memory ∩ top-K-by-round-1-scores; 1.0 =
collapsed to sorted one-shot) and `admit_corr`. Recipe = budget25's
(batch 1024, lr 3e-3, wd 0.05, cosine 300 ep). `--checkpoint_rounds N`
partial activation checkpointing (batch 1024 uncheckpointed OOMs at
~90 GB; ck3 ≈ 62 GB and −14% epoch time vs full ckpt).

1. **R-sweep (R ∈ {1,2,4,7}, K=49, 300 ep each; val rec dropped-only,
   best/final):** R1 0.0989/0.1011 · R2 0.0987/0.0988 · R4 0.0955/0.0986 ·
   R7 **0.0871**/0.0896. Recursion helps monotonically with GROWING
   per-round returns (deltas 1→2 −0.0002, 2→4 −0.0032, 4→7 −0.0084); R7
   is 12% relative better than R1 despite losing ~20 epochs to lr spikes.
2. **R1 final 0.1011 replicates budget25's 0.101 to three decimals** —
   at one shot the rate (49/196) fully determines rec; scorer mechanism
   (learned head vs edge evidence) and encoder access are irrelevant.
3. **The nullity corner did NOT occur**: R7 overlap_r1 fell to ~0.50 —
   half the final memory differs from the one-shot ranking; the recursion
   genuinely re-decides.
4. **Anti-correlation prediction FALSIFIED, informatively.** Round-t+1
   picks land at residual-error percentile 0.31–0.47 (< 0.5): the loop is
   mildly error-AVOIDANT, not error-seeking. Reading: under dropped-only
   loss, memory wants tokens that predict OTHERS; high-self-error patches
   are unpredictable textures that summarize nothing. Admission maps
   corroborate: rough spatial coverage early, late rounds on object
   detail, flat/noisy texture avoided. No contiguous chunk regions.
5. **Open confound (pre-registered, NOT yet separated): depth vs memory.**
   7 weight-shared passes = deeper net regardless of admission quality.
   Queued discriminator: R=7 with RANDOM admission order vs learned.
6. **Trainability**: R7 hit two loss spikes at lr 3e-3 (ep 30, ep 41 →
   0.78), both self-recovered under grad-clip 1.0; R2/R4 smooth. Deeper
   recursion wants a gentler lr.
7. **Probes (attentive, 180 ep, project `asfnet`) — REC AND PROBE
   DIVERGE AS R GROWS.** Top-1, mem49 / all196: budget25 26.5 / 32.0 ·
   R1 28.8 / 35.5 · R2 29.5 / — · R7 **26.3 / 32.6** (R4 probe
   cancelled by user — trend clear). Recursion monotonically improves
   reconstruction yet R7 LOSES ~2.5 pt (memory) and ~3 pt (all-196) vs
   R1; R2 is the probe sweet spot so far. The memory-vs-all gap stays
   ~6 pt at every R — recursion did NOT concentrate class signal into
   memory. Reading: extra weight-shared passes specialize features for
   the decoder (texture/position detail) at the expense of linearly
   separable semantics — the loop optimizes exactly what it's asked to
   and nothing else. Objective, not architecture, is what moves the
   probe (distill/JEPA/fine-tune next if probe numbers are the goal).
8. **Compute-bound correction**: the "85% data-wait" metric was an
   async-CUDA accounting artifact (CPU blocks at the iterator sync point;
   GPU drain lands there). Proof: RAM loader and DALI produce identical
   81 s epochs; nvidia-smi 100%/550 W during epochs. Neocore is
   compute-bound at ~25% MFU; 80 s/epoch is the floor at this size.
   Speed levers left are architectural, not plumbing. (max-autotune ≈
   nil, fused AdamW ≈ nil; the real win was checkpoint_rounds.)
9. **RAM blob cache built + validated** (dataset_ram.py): train 24.9 GB /
   val 1.0 GB uint8 blobs; both cpu and VRAM-resident modes work.
   **Google Drive dataset bank (2026-07-17, DEPLOYED + smoke-proven):**
   bank.py pulls the jpeg cache as ONE 2.35 GiB tar from
   `gdrive:NeocoreBank/jpeg_cache.tar` (a dedicated storage Google
   account, NOT the user's personal Drive; rclone OAuth token =
   RCLONE_DRIVE_TOKEN in secrets.env, forwarded to instances base64 as
   RCLONE_DRIVE_TOKEN_B64). Boot flow in dataset._ensure_cache: local →
   bank → HF build; the 25 GB blob rebuilds locally from the jpegs
   (source images are shorter-side-160 — the jpegs ARE the information;
   the 256 cache upscales). Proven on a virgin instance: pull+extract
   done at +133 s from boot, RUN_COMPLETE at +10m45 for a 1-epoch smoke
   (old HF path: 15+ min to data-ready). Publish/refresh with
   vast/upload_bank.py (256M drive chunks — default 8M chunks stall).
   HARD LESSONS: (a) Vast stop→start re-runs the provision command,
   which `rm -rf`s the repo — a restart WIPED the blobs on the sandbox
   ("stop to preserve state" does not work); onstart now symlinks
   jpeg_cache/ + data/ into /workspace so caches survive same-machine
   restarts. (b) wandb-artifact bank was built then removed same day —
   user is on wandb free plan, didn't want dataset storage there.
   (c) The bank does NOT speed epochs (compute-bound, see item 8) —
   it buys boot time and independence from HF/wandb availability.

Checkpoints/panels in runs/NC_R{1,2,4,7}_K49_300ep/; wandb artifacts
neocore-{i0tqyho2,2wa5npcb,3p04yekp,ab1e8nam}:final. Sweep figure:
runs/night_analysis.png. Full night narrative in POINTS_OF_INTEREST.md.

## Research checkpoint (2026-07-18) — the benchmark verdict: wrong objective, confirmed

Three 300-ep runs completed (artifacts neocore-{bnjvgd3s,tz18glu6,v9k6rg2u}):

1. **Reselect-R7** (`NC_R7_K49_resel_300ep`): best rec 0.1292 / final
   0.1599 — WORSE than R1 (0.0989) despite 7 passes. Depth-vs-memory
   partially resolved: accumulate-R7's gain (0.0871) is not depth — it's
   sticky accumulation; re-picking each round actively hurts and the run
   degraded late (re-stamp norm growth suspected). mem_stability 0.61.
2. **Neocore-AR** (`NCAR_K49_300ep`): best rec 0.1033 ≈ R1. overlap_r1
   0.88 — token-by-token conditioning barely deviates from its own
   one-shot ranking. Strict sequentiality bought nothing.
3. **ConvAE** (`NCONV_K49_300ep`): best rec **0.0516 all-position** — 4x
   better than AE_xattn49's 0.211 at the identical 49x256 rate, zero
   selection machinery. train 72 s/epoch (not faster than transformers:
   early high-res conv stages carry the FLOPs; GPU-bound at 100%).
4. **ConvAE probe (the decisive number): top-1 28.2 / top-5 55.7** —
   dead middle of the 49-token pack (budget25 26.5, R1 28.8, R2 29.5,
   R7 26.3) despite 4x better reconstruction. **Both directions of the
   dissociation are now measured: rec and linear semantics are
   orthogonal under this objective.** All 49-token bottlenecks probe
   26-30% regardless of mechanism; only all-196 pooling exceeds it
   (32-35.5). Pixel reconstruction at fixed rate is rate-determined,
   architecture-blind, and semantics-blind: it cannot see the admission
   machinery this project is about. Next objective candidates (user
   deciding): distillation from a frozen teacher (DINOv2-S), JEPA-style
   latent prediction of dropped tokens, supervised fine-tune. Queued
   discriminator still open: R7 with RANDOM admission order.

Ops (2026-07-18): probes are ~7 s/epoch on a 5090 (21 min / 180 ep,
~$0.11). Boot lottery (`launch --hedge 3`, default) deployed after two
dead boots (m91308 zombie, m137831 second self-stop; both blacklisted):
races N distinct machines, first GATE_PASSED wins, losers destroyed
(~$0.05 each). ConvAE/NeocoreARAE probe path fixed (token_norm d_feat,
Neocore-family isinstance dispatch) — ProbeModel now smoke-tested
locally against every backbone family before cloud runs.

## Research checkpoint (2026-07-18 night) — the vocabulary program opens

Direction change after the benchmark verdict + a day of theory (full
argument in POINTS_OF_INTEREST.md): blocks-first. Chunking = compression
forced through composition over a REUSABLE vocabulary; pixel-loss joint
training could never produce it (per-image rec never pays for reuse; the
free decoder was where the hidden vocabulary leaked). New stack: frozen
DINOv2 blocks -> explicit discrete vocabulary -> selection/arrangement
machinery on top (dumb compositor, code-index targets). Key literature
anchors: DINOSAUR (slots on frozen DINO features work; joint training
degrades), TokenCut/CutLER (chunks are latent in frozen affinities),
I-JEPA (latent targets probe; pixel targets don't), FSQ (VQ stability =
freeze the vocabulary).

**Stage 0-1 DONE (train_vocab.py, run VOCAB_stage01, ~$0.25 total):**
- Token lake: DINOv2-S/14 patch+CLS tokens for all of IN-100 (26 GB
  fp16, 2100 img/s extraction; rebuildable in ~15 min, not banked).
- EMA k-means codebooks K∈{512..8192}: 92-97% perplexity utilization,
  healthy Zipf-ish usage. Figures + codebooks: runs/VOCAB_stage01/ +
  wandb artifact vocab-6duv9qzw.
- **The block-quality gap, measured: DINOv2 patch tokens probe 93.5
  top-1 on IN-100** (attentive, no-aug lake protocol; CLS kNN 92.0)
  vs 35.5 for the best pixel-trained encoder ever built here. ~58 pts
  was block quality, not chunking.
- **Symbolization is semantically nearly free: K=8192 quantized probe
  92.1 (-1.4 pt from raw), K=1024 87.9** — while cosine fidelity is only
  0.73/0.66: geometrically lossy, semantically lossless. 256 symbols x
  13 bits ≈ 416 B/image carries 92% linear class accuracy (~470x vs raw
  fp16 tokens).
- Caveat: lake-probe numbers (no aug, deterministic crop) are not
  comparable to the train_linear_probe.py table; only within-lake
  comparisons are fair.

Stage 2 (designed, not started): Neocore admission machinery over the
lake — memory = K pointers into the codebook, reconstruction = predict
dropped positions' CODE INDICES (classification over vocabulary; no
continuous decoder to leak through). Instruments: admission maps in
code space, assignment-matrix chunk maps, random-admission control,
TokenCut reference parses. Open beyond: recurrence-triggered promotion
(mint new vocab entries for recurring arrangements — online BPE over
parses), episode-structured data (video) as the pressure for caching.

## Research checkpoint (2026-07-19, session close) — the globality law

Classification sweep (train_vocab_cls.py, project neocore-cls; user
stopped it early — result decisive at 4 configs; NOTE: all configs
merged into ONE wandb run 3np8mz3l because run_training.sh exports a
fixed WANDB_RUN_ID and per-config wandb.init resumed it — pop that env
var before reusing the script; per-config numbers recovered by
splitting history on epoch resets):

- CLS_dense (256 tok):    90.52
- CLS_learned_B4_R4:      89.68  (retention_r3 0.78)
- CLS_random_B4_R4:       89.66  (exact tie with learned)
- CLS_learned_B8_R4 (20ep): 89.78 (retention_r3 0.96)

**FOUR tokens ≈ 256 tokens (Δ0.9 pt) and learned == random to 0.02.**
The budget never binds because admission happens over POST-ATTENTION
tokens: 2 shallow blocks (and DINO's 12 before them) make every token a
global summary, so any 4 carry the scene. Selection over summaries is
value-free.

**The session's law, final form: any component that sees everything
dissolves the selection question.** Pixel era: the free decoder. Stage
2: DINO contextualization inflating all arms. Classification: the
shallow encoder. Selection/chunking is only measurable — and only
VALUABLE — when information is forced to flow THROUGH the bottleneck:
admission over LOCAL (pre-mixing) tokens, perception spent only on what
was admitted (foveation, not summarization), and reuse pressure across
episodes so caching pays. Those are the requirements for the next
architecture, whenever it's built.

Positive residue of the day, all reusable: the vocabulary stack (93.5
raw / 92.1 @ 8k codes; codebooks + lake pipeline; ~$0.10-1 experiments
at same-afternoon cadence), the instrument suite (coverage curves,
retention, freq_overlap, census), and one small gem: given the choice,
learned admission spontaneously converged to ACCUMULATE (retention
0.78-0.96) — the model's own answer to reselect-vs-accumulate matches
the pixel-era finding.

Ops residue: boot race (--hedge), Drive-bank gate + download gate,
hardware floors recalibrated from first 5090 scan (221-229 TFLOPS
healthy), live-machine exclusion in offer search, cmd_scan median
crash fixed, scan requires --branch vast-automation until merge.
2026-07-19 network event: US mid-band hosts measured 9-20 mbps to BOTH
HF and Drive (endpoint or DC-side; gate correctly blocks launches
there). m59009/m69554/m139253 all exonerated — every "bad machine"
today was monitoring artifacts or the network event.

## Research checkpoint (2026-07-19) — stage 2 six-arm verdict: random wins, selection machinery loses

train_stage2.py, 6 arms x 3 budgets, ~$5 total, figure
runs/s2_sixarm_analysis.png. Task: admit under budget (positions or
codes), then classify code ids at hidden positions (K=2048 CE; NO
mixing channel — memory is only chosen code embeddings, leak-proof).
Best top1 @ coverage (val, own-residue denominator):

- P-random  39.4/44.8/49.7 @ .10/.19/.38  <- best everywhere
- P-uniform 36.7/38.3/47.4 @ same
- P-learned 30.0/31.5/40.7 @ same         <- WORST position arm
- T-random  21.9/31.6/38.9 @ .25/.47/.79
- T-learned 20.4/20.0/20.9 @ .46/.43/.23 (coverage FELL with budget)
- T-freq    18.3/19.0/18.3 @ .66/.82/.94 (flat — counting is dead)

Findings:
1. **P arms share identical coverage per budget -> clean comparison:
   learned admission LOSES to random AND to a fixed grid at every
   budget.** No denominator excuse. The gate-whisper gradient (only
   final-round scores, via a (1+sigmoid) gate) does not teach selection.
2. **Mask-diversity curriculum confound identified**: the random policy
   re-samples masks every batch (MAE-style augmentation) -> trains a
   better predictor; uniform/learned give near-fixed masks. Explains
   random>uniform (~2-6 pt); the further uniform>learned gap (~7 pt) is
   learned being actively bad. Own-residue denominators additionally
   flatter low-coverage arms (T-freq's exam = the hard residue).
3. **Positions >> types at matched coverage** (T-learned B32 CE 3.78 @
   .23 vs P-random B49 CE 2.03 @ .19): admitting a code covers ALL its
   instances — massively redundant coverage (70 sky patches ~= 5).
   Type admission as "cover full support" wastes budget; a type+exemplar
   variant would fix this.
4. Echoes MAE-literature "random masking beats structured" — harness
   validated against a known result, in symbol space.
5. **User's pre-registered skepticism vindicated**: on masked-symbol
   compression, learned selection does not help. Not yet proven dead:
   the two confounds + broken credit assignment leave one clean rescue
   protocol — FROZEN-PREDICTOR: train one predictor on random masks,
   freeze it, then train/evaluate selection policies against it (kills
   the curriculum confound, enables a greedy-oracle upper-baseline and
   counterfactual value targets for the scorer). That is the next
   experiment if selection gets another shot; else pivot the objective
   (probe-based / distill) per the standing modularity of the harness.

## Local environment (Windows)

- No `python` on PATH. The project env is the `ToastEnv` conda env:
  `& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe"` and
  `& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\Scripts\vastai.exe"`.
- Training does NOT run locally — it runs on rented Vast.ai GPUs.

## Vast.ai cloud training (see vast/README.md for full runbook)

Everything is driven by `vast/launch.py`. **Offer selection is a judgment
call by the operating agent**: run `search`, apply `vast/OFFER_JUDGEMENT.md`
(price near the middle of the range, known-bad machine list), then
`launch --offer <ID>`. Auto-pick only as a fallback.

**Fleet policy (user directive 2026-07-17): GPU profiles** — `launch.py
--profile {5090,6000,b200}` sets gpu_name, price cap, AND the matched
training recipe (batch/lr/data placement/compile/checkpointing; explicit
`--train-args` replaces it wholesale). `5090` is the default workhorse
(overnight runs, **hard cap $0.38/hr**): batch 256, lr 7.5e-4
(linear-scaled from 1024/3e-3), `--data ram` blobs in **system RAM**
(25 GB doesn't fit 32 GB VRAM — never `--data_device cuda` on 5090s),
compile default, ck3; ~2x PRO-6000 epoch time (~160 s R7-class). `6000`
= the proven batch-1024/3e-3 VRAM-resident recipe for user-requested
same-day results ($1.2/hr cap). `b200` = user-approved-only rapid runs
(floor ~$6.9/hr). EPYC hosts fine on 5090s (RAM loader is GPU-bound);
cpu_ram ≥ 48 GB required everywhere. Batch-256 runs are not strictly
comparable to the batch-1024 series — compare within a recipe.

```powershell
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py status
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py launch --smoke
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py launch   # 300-epoch AE run
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py logs
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py destroy  # "kill it"
```

- Secrets live in `vast/secrets.env` (gitignored — NEVER commit; the repo is
  public). Instance state in `.vast/instances.json`.
- Instances health-gate themselves at boot (`vast/benchmark.py` vs
  `vast/thresholds.json`) and self-destroy when unhealthy. Failed runs keep
  the instance alive for inspection.
- **Run persistence (settled 2026-07-16):** runs write checkpoints + eval
  PNGs to `runs/<run_name>/` on the instance (`runs/LATEST` points at the
  newest; `runs/` is gitignored locally too). On success the final
  checkpoint artifact is uploaded to wandb and **VERIFIED committed**
  (`art.wait()`), and only then does the instance self-destroy. If the
  verified upload fails (e.g. a wandb storage outage — 2026-07-15 one ate a
  run's weights under the old async upload), the instance stays alive
  printing `AWAITING_PULL`; fetch with `launch.py pull --id <ID>` (scp;
  account ssh key), then `destroy`.
- The instance clones this repo from GitHub, so cloud-side changes
  (onstart.sh, thresholds, train args defaults) only take effect after push.
