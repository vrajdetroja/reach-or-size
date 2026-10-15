# Ambiguities

Details the paper (arXiv 2502.09263v3) leaves open, or states in a way that admits more than one implementation. Each row records the reading the blind reimplementation starts from and how it gets resolved. A row closes only after the code diff, and any row where the choice is material gets a measured effect in `DISCREPANCIES.md`.

"Allowed reading" means sources other than the GNNPlus repository: the paper, GraphGPS, Tönshoff et al. (2023), LRGB, OGB.

## Resolution method before the code diff: parameter-count matching

Tables 9 and 13 give an exact parameter count for every tuned config (for example 507,351 for GCN+ on Peptides-func). Each combination of readings for A1, A2, A7, A8, A10, A11 and A12 predicts an exact count. Before opening the GNNPlus repository, we will instantiate our implementation under each combination and record which ones match all six in-scope counts. A reading that matches all six is strong evidence; a reading that matches none is ruled out. We commit the result of this step before the code diff, so the diff also tests how well the method worked.

## Result of parameter-count matching (2026-10-15, GNNPlus repository still unopened)

Script: `python -m src.param_count`. Output: `results/param_count_matching.md`. Analytic counts are checked against instantiated models, and `tests/test_models.py` checks every run config against its reported count.

We searched 768 architecture readings for GCN+ and 48 for GatedGCN+. Each reading was combined with 7 head shapes and 7 RWSE widths per dataset, and had to hit all three in-scope counts for its model exactly.
- **GCN+:** exactly one reading matches all three. A wider generic search found the same single solution. It allowed any per-layer cost of the form a·d² + b·d, an edge part of up to d² + 19d, FFN widths 1/2/4 with or without BatchNorm, and RWSE widths 4–64.
- **GatedGCN+:** two readings match, and they differ only in FFN width. No in-scope GatedGCN+ config uses the FFN, and GCN+ molhiv fixes the width at 2d.
- **Tönshoff's GCN:** our rebuild has 485,741 parameters, matching the "486k" in their paper. This is a check that the method works.

This is evidence, not proof. A reading outside the enumerated space could also match.

| # | Reading the counts select | Blind default before matching | Paper text it conflicts with |
| --- | --- | --- | --- |
| A1 | **The GCN+ edge term has no per-layer parameters.** Edges enter through one bond embedding shared by all layers. A learned self-loop edge embedding is ruled out. | Per-layer W_e (d² + d per layer) | **Eq. 6 names W_e^l, "the trainable weight matrix in layer l".** First candidate for `DISCREPANCIES.md`. |
| A7 | GraphGPS-style RWSE encoder: BatchNorm on the K raw return probabilities, Linear(K → p) with **p = 28 on Peptides and 16 on molhiv**, concatenated to an atom embedding of width d − p | One W_PE on [atom embedding ‖ RWSE] | Eq. 12 is consistent only if W_PE is block-diagonal |
| A8 | GatedGCN+ = Dwivedi-style layer: five biased linear maps, edge into the gate, gates normalized over neighbors, edge states updated with their own BatchNorm. Eq. 4 readings without edge updates are ruled out. | Same | Eq. 4 shows no edge term and no gate normalization |
| A10 | **Prediction heads are deep.** GCN+: 4-layer MLP (Peptides-func), 3-layer MLP (Peptides-struct), 4 linear layers halving 256 → 128 → 64 → 32 → 1 (molhiv). GatedGCN+: 3-layer MLP on both Peptides datasets, 3 halving layers on molhiv. | Not chosen | The paper does not mention head depth. Tönshoff et al. found it is the largest single factor on Peptides-struct, so it likely carries part of the GNN+ gain. |
| A11 | OGB encoders with the current atom vocabulary (174). The bond encoder is built **even when the edge module is off** (GCN+ Peptides-struct), as in Tönshoff's GCN. | OGB encoders | None; harmless dead parameters |
| A12 | FFN width 2d, with BatchNorm, exactly as Eq. 10 is written | 2d | None |
| — | GCN has a bias after aggregation | — | Eq. 2 shows no bias |

**Still open after matching (equal parameter counts):**
- A1 sub-readings: edge embedding added after W, which is what we run, vs. (h_u + e_uv)W, implemented as `shared_pre_linear`; self-loop edge term zero vs. absent.
- A2, A3, A4, A13–A17, and the head's activation and dropout.

These go to the code diff.

## Table (pre-matching record)

| # | Detail | Candidate readings | Blind default | Resolve by |
| --- | --- | --- | --- | --- |
| A1 | **Where the edge term enters Eq. 6.** The equation puts `e_uv W_e` inside σ(·) next to the normalized sum, but as written it is not clearly inside the sum over u, even though it depends on u. The sum also runs over N(v) ∪ {v}, and the self-loop has no edge feature `e_vv`. | (a) Per-message, inside the sum, GINE-style: Σ_u [norm·h_u W + e_uv W_e]. (b) Aggregated separately: Σ_u norm·h_u W + Σ_{u∈N(v)} e_uv W_e. Self-loop: (i) zero edge feature, (ii) a learned self-loop embedding, (iii) no edge term for u = v. | (a) + (i) | Parameter count, then code diff; measure if material |
| A2 | **Degree normalization of the edge term** | `e_uv W_e` scaled by 1/√(d̂_u d̂_v) like `h_u`, or added raw | Raw, as Eq. 6 is typeset | Code diff; measure if material |
| A3 | **LR schedule.** Tables give warmup epochs (5 for all in-scope configs) but no decay. | Cosine decay after linear warmup (GraphGPS `cosine_with_warmup`); constant after warmup; reduce-on-plateau (LRGB default) | Cosine with warmup | Code diff |
| A4 | **Which epoch's test score is reported** | Test at best-validation epoch (GraphGPS/OGB convention); last epoch | Best validation epoch | Code diff. This alone can move AP by about 0.01. |
| A5 | **Size of the hyperparameter search.** Sec. 4: "we do not perform an exhaustive search". | Unknown number of configs per dataset | — | Not reported. The report states it as unknown. The public repo's configs show only the final settings. |
| A6 | **Retrained transformer numbers.** Sec. 4 says GTs were retrained in the same search space but did not beat their published numbers, so the published ones are shown. | Not reported | — | Not reported. We compare only against published transformer numbers and say so. |
| A7 | **RWSE details (Eq. 12: `x_v = [x_v ‖ x_v^RWSE] W_PE`)** | Walk steps 1..K (K = 32 Peptides, 20 molhiv) or 2..K. Raw RWSE normalized (GraphGPS applies BatchNorm to raw RWSE) or not. `x_v` raw categorical atom features, or already embedded by an atom encoder. One linear map on the concatenation (as typeset) vs. the GraphGPS split (atom embedding of width d − d_pe, concatenated with a linear RWSE embedding of width d_pe). | Steps 1..K; BatchNorm on raw RWSE; embed atoms first, then one linear map on the concatenation | Parameter count, then code diff |
| A8 | **GatedGCN+ "edge feature module."** Eq. 4 has no edge term in the gate, yet Table 13 turns the edge module on for every config. | (a) Dwivedi/Bresson gating with edge input, η = σ(A h_v + B h_u + C e_uv), with edge states updated each layer. (b) Eq. 4 plus an additive `e_uv W_e` message term as in Eq. 6. (c) (a) without per-layer edge updates. | (a) | Parameter count, then code diff |
| A9 | **Source of the vanilla GCN numbers.** *Largely resolved from the paper:* Table 3 baselines are cited from original papers, and 0.6860 ± 0.0050 is Tönshoff et al.'s tuned GCN (their Table 2: 6 layers, hidden 235, head depth 3, dropout 0.1, RWSE, batch 200, 250 epochs, lr 0.001, 486k params). Molhiv 0.7606 ± 0.0097 is the OGB paper's GCN. | Still open: whether Tönshoff's GCN layer has residuals or BatchNorm, its activation, and its RWSE walk length | Read Tönshoff's config files (allowed reading) | Allowed reading, before the environment checks |
| A10 | **Prediction head g(·).** Sec. 2 calls it "a prediction head"; depth, width and activation are not given. Tönshoff et al. found head depth is the largest single factor on Peptides-struct. | Linear; 2-layer MLP; 3-layer MLP (Tönshoff's GCN); GraphGPS `san_graph` head with halving widths | Pick by parameter count | Parameter count, then code diff; measure if material |
| A11 | **Input encoders.** Peptides and molhiv both use 9 categorical atom and 3 categorical bond features. | OGB AtomEncoder/BondEncoder (sum of per-feature embeddings); one-hot plus linear | OGB encoders | Parameter count, then code diff |
| A12 | **FFN width and the skip inside Eq. 10.** FFN(h) = BN(σ(h W₁) W₂ + h). Hidden width of W₁ is not given. Eq. 10 also contains its own skip connection and BatchNorm. | Width 2d (GraphGPS) or d | 2d | Parameter count, then code diff |
|  | *Consequence:* GCN+ on molhiv has "Residual Connections: False" and "FFN: True", so per Eq. 10 it still has a skip connection, inside the FFN. "No residuals" means no skip around message passing only. | — | Implement Eq. 10 as written | Code diff |
| A13 | **Activation σ** | ReLU (the example after Eq. 2); GELU (some LRGB configs) | ReLU | Code diff |
| A14 | **Optimizer details not in the tables** | Gradient clipping on/off; AdamW β and ε; evaluation every epoch or less often | No clipping; PyTorch defaults; evaluate every epoch | Code diff |
| A15 | **Losses and metric implementations** | Peptides-func: BCE-with-logits, AP averaged over the 10 tasks (LRGB). Peptides-struct: L1 loss, MAE over 11 targets. molhiv: BCE, ROC-AUC via the OGB Evaluator. | As listed (standard protocols) | LRGB/OGB code (allowed reading); code diff |
| A16 | **Dropout placement beyond Eq. 8.** Eq. 8 puts dropout after the activation in each layer. Input dropout and dropout inside the head or FFN are not mentioned. | None elsewhere; dropout also in the head (GraphGPS default) | Only where Eq. 8 puts it | Code diff |
| A17 | **Data pipeline details** | Shuffling, `drop_last`, and RWSE computed on the graph as stored (both edge directions) | Shuffle on, `drop_last` off, RWSE on the undirected graph | Code diff |

## Readings that are not ambiguous but are easy to get wrong

- Eq. 11 adds the residual before the FFN: h^l = FFN(Dropout(σ(BN(·))) + h^{l−1}). With residuals off and FFN off (all Peptides GCN+ configs), a layer is Dropout(σ(BN(GCN-with-edges))).
- BatchNorm sits before the activation (Eq. 7) and dropout after it (Eq. 8).
- Peptides GCN+ uses neither residuals nor FFN, and Peptides-struct GCN+ also has no edge features (see `CLAIMS.md`).
- A dash in Tables 5–6 means the component was off in the tuned config, not that removing it had no effect.
- RWSE-K is built from K-step return probabilities. A closed walk of length k reaches at most ⌊k/2⌋ hops from its start, so RWSE-32 carries structure from up to 16 hops away and RWSE-8 from up to 4. This matters for the extension (`PREREGISTRATION.md` §5).
