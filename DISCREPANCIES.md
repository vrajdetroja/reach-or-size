# Discrepancies: paper vs. released code

- Repository first opened at our commit: `58da50d` on 2026-10-19. This is after the blind implementation and the parameter-count matching (`e74ab83`).
- GNNPlus commit inspected: `0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28` (2025-08-19), github.com/LUOyk1999/GNNPlus.
- Parameter-count matching result committed at: `e74ab83` (see `AMBIGUITIES.md`).

Their code is GraphGPS/GraphGym. The in-scope configs are `configs/{gcn,gatedgcn}/{peptides-func,peptides-struct,hiv}.yaml`. File references below are relative to their repository.

## Paper vs. code

| # | Ambiguity | Paper says | Code does | Our blind reading | Material? | Measured effect (3 seeds, GCN+ Peptides-func) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A1, A2 | Eq. 6: Σ over N(v) ∪ {v} of [h_u W / √(d̂_u d̂_v) + e_uv W_e^l], with a per-layer W_e | `gcne` layer (`layer/gcn_conv_layer_e.py`): **no self-loops, no degree normalization, ReLU inside the message**. Each layer computes h_v ← Σ_{u ∈ N(v)} ReLU(h_u W + e_uv) + b, where e_uv is one bond embedding shared by all layers. Used by GCN+ on Peptides-func and molhiv. GCN+ on Peptides-struct uses PyG's standard `GCNConv` (normalized, with self-loops, no edges), as in Eq. 2. | Eq. 6 as typeset with the shared bond embedding: normalized with self-loops, edge term added after W, no ReLU | **Yes.** With residuals off, as on Peptides-func, a node's own state never enters its update. | **−0.0054 ± 2·SE 0.0102: not material** (0.7116 ± 0.0088, 3 seeds). The one-seed gap of 0.066 in the environment checks was the paper-literal model as a whole, mostly rows 3+4, not this row. |
| 2 | A13 | σ, "e.g. ReLU" | **GELU** in the GCN layers, FFN and heads on Peptides; ReLU on molhiv. The GatedGCN layer's own activation is **always ReLU** (hard-coded default); its head uses the config's activation. | ReLU everywhere | Possibly | **−0.0074 ± 2·SE 0.0062: material, small** (0.7095 ± 0.0052) |
| 3 | A16 | Dropout after each layer's activation (Eq. 8) | The MLP head (`head/mlp_graph.py`, Peptides) applies **dropout before every Linear** with the GNN's rate. The FFN has dropout after its activation and after its second Linear. The halving head (molhiv) has none. | Dropout only where Eq. 8 puts it | Possibly | **With #4: −0.0437 ± 2·SE 0.0154, material and the largest single factor** (0.6733 ± 0.0132) |
| 4 | A14 | Unstated | **Gradient-norm clipping at 1.0** in every in-scope config | No clipping | Possibly | Measured together with #3 (see row 3) |
| 5 | A12 | Eq. 10: FFN(h) = BN(σ(hW₁)W₂ + h) | **An extra BatchNorm before the FFN**: h ← BN₂(BN₁(h) + Drop(Drop(σ(BN₁(h)W₁))W₂)). Width 2d. | Eq. 10 as written, width 2d | Probably small. Only GCN+ molhiv uses the FFN. | Not measured (molhiv only) |
| 6 | A10 | Head not described | Peptides: MLP head with `layers_post_mp` linear layers at width d (GCN+ func 4, others 3). molhiv: halving head that **ignores `layers_post_mp` and always has 3 linear layers** (256 → 128 → 64 → 1). | Count-matched: MLP-4 / MLP-3 ✓. Halving-3 for GatedGCN+ ✓. **Halving-4 for GCN+ molhiv ✗** (see the note below). | Small | Not measured |
| 7 | A7 | Eq. 12: [x ‖ RWSE] W_PE | GraphGPS encoder: BatchNorm on RWSE steps 1..K, then Linear(K → 28) on Peptides or Linear(K → 16) on molhiv, concatenated to a (d − p)-wide atom embedding | Count-matched: identical ✓ | No | — |
| 8 | A8 | Eq. 4 shows no edge term | Dwivedi-style layer: five biased linears, edge input to the gate, gates normalized over neighbors, edge states updated with their own BatchNorm. The edge residual follows the node residual flag. | Count-matched: identical ✓ | No | — |
| 9 | A11 | — | GraphGym Atom/Bond encoders (OGB vocabularies, 174 atom entries). The bond encoder is built even with the edge module off (`gcn` on Peptides-struct). | Count-matched: identical ✓ | No | — |
| 10 | A3 | 5 warmup epochs, decay unstated | GraphGPS `cosine_with_warmup`, stepped per epoch, 1e-6 floor during warmup | Identical ✓ | No | — |
| 11 | A4 | Unstated | Test score at the best-validation epoch (`train/custom_train.py`) | Identical ✓ | No | — |
| 12 | A15 | Standard protocols | Multilabel BCE-with-logits, L1 for Peptides-struct, OGB ROC-AUC, LRGB-style AP | Identical ✓ | No | — |
| 13 | A17, data | Standard splits | **A different pipeline from ours.** Their code downloads the LRGB SMILES CSV and the stratified-split pickle (md5-checked) from Dropbox and featurizes with OGB `smiles2graph` (rdkit). We use PyG's preprocessed LRGB files. Both derive from the same LRGB release. | Standard LRGB/OGB splits | **No.** Both pipelines give byte-identical Peptides-func graphs in every split (SHA-256 over node features, edges, edge features and labels). | — |
| 14 | Seeds | 5 seeds per number | `--repeat` gives seeds `seed, seed+1, ...`. The README's launch commands use `--repeat 4` for Peptides and `--repeat 2` for molhiv. | 3 seeds (preregistered) | Not a code difference. Recorded because the paper reports 5. | — |

## What parameter-count matching got right and wrong

Before the code was opened, every reading it selected held except one:
- no per-layer edge weights, with one shared bond embedding;
- the GraphGPS RWSE encoder, including its widths of 28 and 16;
- the Dwivedi gated layer with edge updates;
- the bond encoder built even with the edge module off;
- a bias after aggregation;
- the current OGB atom vocabulary;
- FFN width 2d;
- 5 of the 6 prediction heads.

**The exception is instructive.** For GCN+ molhiv, it chose a 4-layer halving head with one FFN BatchNorm. The code has a 3-layer head and two FFN BatchNorms. Both give exactly 1,407,641 parameters, because the fourth head layer costs 2,080 − 32 = 2,048 parameters and one extra BatchNorm per layer costs 2d × 4 = 2,048. Our search space did not include the second BatchNorm, so the tie was resolved the wrong way.

The method cannot see choices that cost no parameters: the self-loops, normalization and ReLU in row 1, the activation, dropout and clipping. Those are exactly the material rows.

## Measurements (2026-10-21, RTX 4090, 3 seeds each; `results/discrepancy.md`)

Baseline: the code-aligned C1 runs, 0.7170 ± 0.0012 test AP (paper 0.7261 ± 0.0067). Each row puts one factor back to the paper's reading.

| Row | Change from the code-aligned model | Test AP | Effect | 2·SE | Material? |
| --- | --- | --- | --- | --- | --- |
| 1 | Edge message as Eq. 6 is typeset (normalized, self-loops, no ReLU) | 0.7116 ± 0.0088 | −0.0054 | 0.0102 | no |
| 2 | ReLU instead of GELU | 0.7095 ± 0.0052 | −0.0074 | 0.0062 | yes, small |
| 3+4 | No head dropout, no gradient clipping | 0.6733 ± 0.0132 | **−0.0437** | 0.0154 | **yes** |
| all | All of the above: the model the paper's text describes | 0.6462 ± 0.0041 | **−0.0708** | 0.0050 | **yes** |

**Reading.**
- **The text alone loses about 0.07 AP.** A model built from the text is 0.071 AP below the released code, about 10.6 of the paper's own seed standard deviations, and far outside the C1 tolerance.
- **Most of the loss comes from the head.** Rows 3+4 cover regularization the paper never mentions: dropout before every layer of the 3–4-layer prediction head, and gradient clipping at 1.0. Without them the head tends to overfit early. Two of the three paper-literal seeds hit their best validation epoch at 48 and 60, as did two of the three no-head-dropout seeds (30 and 39), against 145–184 for the code-aligned model. The third seed in each group peaked late (167 and 253).
- **The edge message doesn't matter here.** The difference between the typeset Eq. 6 and the released code's message, the most visible one in the code diff, costs only 0.005 on its own.
- **The factors compound.** The single-row effects add to −0.057, against −0.071 when all are combined.

Rows 5 and 6 (FFN BatchNorm, molhiv head depth) affect only molhiv and were not measured. The code-aligned configs follow the code on both.
