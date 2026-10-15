# Ambiguities

Details the paper (arXiv 2502.09263v3) leaves open, or states in a way that admits more than one implementation. Each row records the reading the blind reimplementation starts from and how it gets resolved. A row closes only after the code diff, and any row where the choice is material gets a measured effect in `DISCREPANCIES.md`.

"Allowed reading" means sources other than the GNNPlus repository: the paper, GraphGPS, Tönshoff et al. (2023), LRGB, OGB.

## Resolution method before the code diff: parameter-count matching

Tables 9 and 13 give an exact parameter count for every tuned config (for example 507,351 for GCN+ on Peptides-func). Each combination of readings for A1, A2, A7, A8, A10, A11 and A12 predicts an exact count. Before opening the GNNPlus repository, we will instantiate our implementation under each combination and record which ones match all six in-scope counts. A reading that matches all six is strong evidence; a reading that matches none is ruled out. We commit the result of this step before the code diff, so the diff also tests how well the method worked.

## Table

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
