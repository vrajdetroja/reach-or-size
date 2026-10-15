# Preregistration

> **Binding from the commit that adds this file (2026-10-15),** which precedes any code in `src/`. Later changes are logged in §7 with their dates.

## 0. Design decisions

| # | Decision | Choice | Why |
| --- | --- | --- | --- |
| D1 | Arm A (training size): how training length scales when the training set shrinks | **Fixed optimizer steps.** Every cell gets the 204,000 steps of the full-data run. | With fixed epochs, the 25% model gets 4× fewer updates. If RWSE speeds up learning, that alone inflates the PE gain at small sizes and biases the test toward H-size. |
| D2 | Arm C (walk length): which K values | **K ∈ {6, 16, 32}** | At L = 3 layers, RWSE-6 is the longest walk whose structure lies entirely inside the message-passing receptive field (⌊6/2⌋ = 3 hops). RWSE-8 already reaches 4 hops, which blurs the contrast. |
| D3 | A third hypothesis, H-local, and SE-based decision rules (§5.5) in place of a fixed "flat within ±0.01" criterion | **Yes** | "Flat within ±0.01" cannot be shown with 3 seeds: the 90% CI on a difference of gains is ±0.015 wide. Without H-local, "neither arm moves" has no interpretation. |

## 1. Scope

| | In scope | Out of scope |
| --- | --- | --- |
| Datasets | Peptides-func, Peptides-struct, ogbg-molhiv | The other 11 |
| Models | GCN+, GatedGCN+; vanilla GCN (Tönshoff config) as the Peptides-func baseline; OGB GCN as the molhiv baseline | GIN+, any transformer |
| Seeds | 3 per reported number (seeds 0, 1, 2) | The paper's 5 |
| Comparison | Our numbers vs. the paper's GNN+ numbers and the competitor numbers it cites | Retraining competitors |
| Efficiency | s/epoch reported as a side note | Not adjudicated |

The verdict can say whether the claims hold on these three datasets, which cover long-range and molecular tasks. It cannot say anything about the 14-dataset ranking. The abstract will state this limit.

## 2. Blindness protocol

1. The GNNPlus repository (github.com/LUOyk1999/GNNPlus, previously `tunedGNN-G`) stays unopened until three things are true: the blind implementation passes CPU smoke tests for all three models, RWSE is cached, and the parameter-count matching result (`AMBIGUITIES.md`) is committed.
2. Allowed reading before then: the paper, GraphGPS, Tönshoff et al. (2023) and their configs, the LRGB and OGB papers and code, PyTorch Geometric.
   Only public sources are used.
3. The commit hash at which the repository is first opened goes in `DISCREPANCIES.md`.
4. **Limits of blindness.** Blind means the repository is not opened. It cannot rule out earlier exposure to the public repository, and the report says so. Implementation choices are checked against the paper's equations and its parameter counts, not against recollection of the code.

## 3. Reproduction criteria

**3.1 Tolerance.** A per-number claim (C0–C4 in `CLAIMS.md`) **reproduces** if |our 3-seed mean − paper mean| ≤ 2 × the paper's reported std.

**3.2 Outcomes.** Each claim gets one of three verdicts.
- **Reproduced:** inside tolerance.
- **Direction holds, magnitude differs:** outside tolerance, but the GNN+ model still beats its baseline (C0/C0b) by more than 2 × the SE of the difference.
- **Not reproduced:** anything else, and only after the failure checks in §3.3.

**3.3 Failure checks.** Before any claim is called "not reproduced", three checks must be done:
1. Every `DISCREPANCIES.md` row marked material has been measured.
2. Their code has been run on the same claim as a control (at least 1 seed).
3. Dataset split sizes and file hashes have been checked against the standard LRGB/OGB releases.

If their code reproduces and ours does not, the problem is our implementation, and we report the cause rather than a failure.

**3.4 PE claims (C5a, C5b).** Reproduced if our drop (with RWSE − without) > 2·SE, where SE = √(s_on²/3 + s_off²/3) from our own seeds. Expected power at the paper's effect sizes: 0.82 for C5a, about 1.0 for C5b.

**3.5 Rankings (R1–R3).** We use the distinguishability rule in `CLAIMS.md`. On ogbg-molhiv, if the gap to any competitor is within that rule, the wording is "indistinguishable from [models]". We do not write "confirms" or "fails to reproduce" the rank.

**3.6 Environment checks.** Two checks run before any headline result:
1. **Baseline.** Tönshoff-config GCN on Peptides-func, 3 seeds, our implementation. Pass if the mean is in [0.6760, 0.6960].
2. **Harness.** GCN+ on Peptides-func, 1 seed each with their code and ours. Their run must land in [0.7127, 0.7395]. If it does not, our data pipeline or package versions differ from theirs, and that gets fixed before anything else.

## 4. Runs and environment

- Seeds 0, 1, 2 for every cell. Cells that differ only by PE share a seed, so data order is paired. The primary analysis still treats cells as independent, which is the conservative choice; the paired analysis is reported as secondary.
- Headline numbers and the extension run in one environment: an NVIDIA RTX 4090, with package versions pinned and recorded.
- We report the test score at the best-validation epoch. If A4 turns out differently in their code, we report both, and the paper's choice becomes the headline.
- Every run logs per-epoch train/val/test metrics to CSV, collected in `results/all_runs.csv`.

## 5. Extension: why does RWSE help?

### 5.1 Hypotheses

- **H-size (the paper's Observation 6).** RWSE matters on small datasets because a model trained on little data cannot learn graph structure on its own. The paper offers this without a test. Its small and large datasets also differ in graph type, so size is confounded.
- **H-reach (ours).** RWSE-K summarizes closed walks of length up to K, which carry structure from up to ⌊K/2⌋ hops away. The tuned GCN+ for Peptides-func has 3 layers, so message passing sees 3 hops, while RWSE-32 sees 16. On this reading, RWSE helps because it carries structure from beyond the receptive field.
- **H-local (added).** RWSE helps because it encodes local structure that message passing cannot compute at any depth tested, such as the size of the ring a node sits on (5- vs 6-rings in proline and aromatic side chains). OGB atom features include an is-in-ring flag but not the ring size.

### 5.2 Quantities

- g(·): the PE gain, i.e. the mean test metric with RWSE minus without, at the stated condition, from our own seeds.
- Δ̂₀: our measured gain for the tuned GCN+ Peptides-func config at full data. These are the same cells as C5a, so Δ̂₀ = ĝ(100%) = ĝ(K = 32, L = 3). Thresholds use Δ̂₀, not the paper's 0.0192.
- **Precondition.** If C5a is not reproduced, the gain we would study is not detectable. In that case no extension run starts until a written amendment is committed (§7).

### 5.3 Arms

Everything not listed stays at the tuned GCN+ Peptides-func config: 3 layers, edge features on, BatchNorm, dropout 0.2, residuals off, FFN off, hidden 275, mean pooling, batch 16, lr 0.001, 300 epochs, 5 warmup epochs, weight decay 0.

| Arm | What changes | Cells (× 3 seeds) | Reused from reproduction |
| --- | --- | --- | --- |
| A. Training size | Train on nested random subsets of 25%, 50% and 100% of the training split (2,718 / 5,436 / 10,873 graphs), with one fixed subset seed shared by all model seeds. Val and test are unchanged. **Fixed optimizer steps (D1):** 204,000 steps per run (300 × 680), evaluated every 680 steps, warmup 3,400 steps, cosine schedule over the rest. | {25, 50, 100}% × {RWSE-32, none} = 6 | Both 100% cells (C1 and C5a) |
| B. Depth | L ∈ {3, 6}, **residuals on at both depths**, so that we measure reach rather than trainability (paper App. B.1). Hidden width is set so the RWSE-on model is within ±2% of 507,351 parameters. The same width is used for PE-off. | {3, 6} × {RWSE-32, none} = 4 | None, because residuals change the base config |
| C. Walk length | RWSE-K for K ∈ {6, 16, 32} (D2), at L = 3 | {6, 16, 32, none} = 4 | K = 32 (C1) and none (C5a) |

### 5.4 Predictions

| Arm | Primary contrast X | H-size predicts | H-reach predicts | H-local predicts |
| --- | --- | --- | --- | --- |
| A | X_A = g(25%) − g(100%) | X_A ≥ 0.5·Δ̂₀ (gain at 25% ≥ 1.5·Δ̂₀) | X_A ≈ 0 | X_A ≈ 0 |
| B | X_B = g(L=3) − g(L=6) | X_B ≈ 0 | X_B ≥ 0.5·ĝ(L=3) | X_B ≈ 0 |
| C | X_C = mean(K=32) − mean(K=6) = g(32) − g(6) | No prediction | X_C ≥ 0.5·Δ̂₀ (most of the gain needs walks longer than 2L) | X_C ≈ 0 with g(6) > 0 (short walks carry the gain) |

### 5.5 Decision rules

For each arm, SE is computed from our seeds with cells treated as independent, and m is the smallest effect the hypothesis predicts (0.5·Δ̂₀, or 0.5·ĝ(L=3) for arm B).

- **Positive:** X − 2·SE > 0.
- **Absent:** X + 1.645·SE < m, i.e. the one-sided 95% upper bound rules out the predicted effect.
- **Inconclusive:** anything else.

Joint reading:

| Pattern | What we write |
| --- | --- |
| A positive, B and C not positive | Evidence for H-size |
| B or C positive, A absent | Evidence for H-reach; the size explanation does not hold on Peptides-func |
| C absent with g(6) − 2·SE(g(6)) > 0, B not positive | Evidence for H-local: the gain comes from local structure message passing does not compute |
| A positive and (B or C) positive | RWSE does more than one job; we quantify each |
| Anything else | Inconclusive with the seeds run; we report effect sizes, CIs and the seeds needed |

Every row is reportable. A wrong prediction stated in advance is a result.

### 5.6 Statistical power

Inputs: σ ≈ 0.0081, pooled from the paper's per-run stds for GCN+ on Peptides-func with RWSE (0.0067) and without (0.0093); Δ₀ = 0.0192.

| Seeds per cell | SE of one gain (also arm C's X_C) | SE of a difference of gains (arms A, B) | Power, X = 0.5·Δ₀ | Power, X = Δ₀ (gain fully disappears) |
| --- | --- | --- | --- | --- |
| 3 | 0.0066 | 0.0094 | 0.16 | 0.52 |
| 5 | 0.0051 | 0.0072 | 0.25 | 0.74 |
| 6 | 0.0047 | 0.0066 | 0.29 | 0.82 |

Power is for the one-sided "positive" rule (X > 2·SE). Detecting a 50% change in the gain with 80% power would take about 23 seeds per cell.

**What this means.** Arms A and B can detect only large effects, where the gain mostly appears or disappears. "Inconclusive" is the most likely outcome for them, and that is acceptable. Arm C, because X_C is a single difference of means, has power ≈ 0.82 at 3 seeds if H-reach is right and g(6) ≈ 0.

**Extra-seed rule.** After 3 seeds, if an arm's X lies between 1·SE and 2·SE, that arm gets 2 more seeds per cell. At most one arm is extended, B before A. Arm C is already the best powered.

### 5.7 Synthetic control (CPU only)

**Purpose.** Show, on graphs where the answer is known, whether RWSE's benefit tracks reach (L vs. ring size) or local structure that GCN+ cannot compute. A clean result shows only that a mechanism is possible, not that it operates on Peptides.

**Task.** Each graph is a path with one ring of size c attached at a random interior node, with the ring sharing that node with the path. The label is binary: which of two ring sizes was attached. Total node count is drawn from U[40, 70] independently of the label, so size carries no signal.

| Setting | Values |
| --- | --- |
| Ring-size pairs | (4, 6), (8, 10), (12, 14), (16, 18), and the chemically relevant (5, 6) |
| Node features | Constant |
| Edge features | None |
| Data | 2,000 train / 500 val / 500 test graphs per pair |
| Model | GCN+ block: BatchNorm, residuals on, FFN off, hidden 64, mean pooling |
| Depth | L ∈ {1, 2, 3, 4, 6, 8} |
| PE | None, or RWSE-K for K ∈ {4, 8, 16, 32} |
| Seeds | 3 |

Predictions (pair (c, c+2)):
- **H-reach mechanism.** Without RWSE, accuracy stays near chance for L < c/2 and rises once L ≥ c/2. With RWSE-K, accuracy is high whenever K ≥ c, at any L.
- **H-local mechanism.** Without RWSE, accuracy stays near chance even at L ≥ c/2 + 2. That would mean GCN+ cannot compute ring size inside its receptive field. The (5, 6) pair is the one that bears on Peptides.

### 5.8 Prior work this extension must be read against

- **DRew** (Gutteridge et al., ICML 2023), App. B.5, Figs. 4–5. Peptides-func, GCN and residual GCN with 500k parameters across depths, with and without LapPE. This is the closest prior evidence on PE gain versus depth, and it is digitized and discussed before the arms run. Our arm B differs in three ways: RWSE, a local structural encoding, instead of LapPE, a global positional one; the tuned GNN+ regime instead of untuned LRGB baselines; and a joint test against H-size and H-local.
- **Grötschla, Xie & Wattenhofer (2024).** The relative ordering of PEs is stable across the depths and dropouts they test, and RWSE does not beat simpler PEs on Peptides-func. We will check that no claim we make about PE is already in their results.

## 6. What cannot change after this commit

The tolerances (§3), the environment-check thresholds (§3.6), the arms, their cells and seeds (§5.3), the predictions (§5.4), the decision rules (§5.5) and the extra-seed rule (§5.6). Implementation details still open in `AMBIGUITIES.md` may change; that is the point of the diff.

## 7. Deviations

Any change after the commit is logged here with its date, its reason and whether it was made before or after seeing the relevant results.

| Date | Section | Change | Before or after seeing results | Reason |
| --- | --- | --- | --- | --- |
