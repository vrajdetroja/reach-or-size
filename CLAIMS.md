# Claims

**Paper:** Luo, Shi & Wu, *Can Classic GNNs Be Strong Baselines for Graph-level Tasks? Simple Architectures Meet Excellence*, ICML 2025 (PMLR v267). Pinned version: [arXiv 2502.09263v3](https://arxiv.org/abs/2502.09263v3). All table and equation numbers below refer to v3.

**Single claim being tested:** with tuned edge features, BatchNorm, dropout, residuals, FFN and RWSE, classic message-passing GNNs (GCN, GatedGCN) match or beat graph transformers on graph-level tasks. We test it on 3 of the paper's 14 datasets with 2 of its 3 models, so our verdict cannot speak to the 14-dataset ranking.

Target numbers are the paper's (mean ± std over 5 seeds). Tolerance is preregistered in `PREREGISTRATION.md` §3: a claim reproduces if our 3-seed mean lies within 2 × the paper's reported std of the paper's mean.

## Per-number claims

| # | Claim | Evidence | Type | Target | Tolerance (2σ) | Our result |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | Tuned "vanilla" GCN on Peptides-func | Table 3, cited from Tönshoff et al. 2023 | Baseline (environment check) | AP 0.6860 ± 0.0050 | [0.6760, 0.6960] | **0.6925 ± 0.0037** (3 seeds, CPU environment check): **reproduced** |
| C0b | Vanilla GCN on ogbg-molhiv | Table 4, cited from Hu et al. 2020 (OGB) | Baseline | AUROC 0.7606 ± 0.0097 | [0.7412, 0.7800] | — |
| C1 | GCN+ on Peptides-func | Table 3 | Per-number | AP 0.7261 ± 0.0067 | [0.7127, 0.7395] | — |
| C2 | GCN+ on Peptides-struct | Table 3 | Per-number | MAE 0.2421 ± 0.0016 | [0.2389, 0.2453] | — |
| C3a | GatedGCN+ on Peptides-func | Table 3 | Per-number | AP 0.7006 ± 0.0033 | [0.6940, 0.7072] | — |
| C3b | GatedGCN+ on Peptides-struct | Table 3 | Per-number | MAE 0.2431 ± 0.0020 | [0.2391, 0.2471] | — |
| C4a | GCN+ on ogbg-molhiv | Table 4 | Per-number | AUROC 0.8012 ± 0.0124 | [0.7764, 0.8260] | — |
| C4b | GatedGCN+ on ogbg-molhiv | Table 4 | Per-number | AUROC 0.8040 ± 0.0164 | [0.7712, 0.8368] | — |
| C5a | Removing RWSE hurts GCN+ on Peptides-func | Table 6 | Ablation | 0.7261 → 0.7069 (drop 0.0192) | Our drop > 2 × SE of our difference | — |
| C5b | Removing RWSE hurts GCN+ on ogbg-molhiv | Table 6 | Ablation | 0.8012 → 0.7593 (drop 0.0419) | Our drop > 2 × SE of our difference | — |
| C6 | RWSE matters because the dataset is small (Observation 6) | Sec. 5.2, offered without a test | Explanation, tested by the extension | — | Preregistered in `PREREGISTRATION.md` §5 | — |
| C7 | GNN+ trains faster per epoch than GraphGPS | Tables 2–4, 8–13 | Not tested | — | — | Not tested: different hardware |

## Ranking claims

These are softer than the per-number claims. We can say whether our 3-seed mean keeps the rank against the numbers the paper cites, and whether the gap is larger than seed noise. We cannot retrain any competitor.

| # | Claim | Nearest competitors as cited in the paper | Note |
| --- | --- | --- | --- |
| R1 | GCN+ ranks first on Peptides-func | DRew 0.7150 ± 0.0044, GRED 0.7133 ± 0.0011, GSSC 0.7081 ± 0.0062 | Gap to second is 0.0111 AP, about 1.7 of the paper's own std. |
| R2 | GCN+ ranks first on Peptides-struct | Graph ViT/MLP-Mixer 0.2449 ± 0.0016, MGT 0.2453 ± 0.0025, GPNN 0.2454 ± 0.0003 | Gap to the best non-GNN+ model is 0.0028 MAE. |
| R3 | GatedGCN+ ranks first on ogbg-molhiv (pretrained models excluded) | Subgraphormer 0.8038 ± 0.0192, GSSC 0.8035 ± 0.0142, GCN+ 0.8012 ± 0.0124, Graph ViT/MLP-Mixer 0.7997 ± 0.0102, GECO 0.7980 ± 0.0200 | Gap to second is 0.0002 AUROC. Five non-pretrained models (three GTs, one state-space model and GCN+) sit within 0.006, plus the two pretrained ones (Graphormer 0.8051†, EGT 0.8060†). With 3 seeds this rank cannot be confirmed or refuted; the preregistered wording is "indistinguishable". |

Distinguishability rule (preregistered): a gap counts as real only if |gap| > 2·√(s_ours²/3 + s_comp²/n_comp), with n_comp the seed count stated by the competitor's source paper, or n_comp = 1 when it is not stated.

## What "GNN+" actually is on each in-scope config

Read from Tables 9 and 13. A dash in the ablation tables (Tables 5–6) means the component was off in the tuned config, not that removing it had no effect.

| Config | Edge | BN | Dropout | Residual | FFN | RWSE | Layers | Hidden | Params | Components on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN+ Peptides-func | ✓ | ✓ | 0.2 | ✗ | ✗ | 32 | 3 | 275 | 507,351 | 4 of 6 |
| GCN+ Peptides-struct | ✗ | ✓ | 0.2 | ✗ | ✗ | 32 | 5 | 255 | 506,127 | **3 of 6** |
| GCN+ ogbg-molhiv | ✓ | ✓ | 0.1 | ✗ | ✓ | 20 | 4 | 256 | 1,407,641 | 5 of 6 |
| GatedGCN+ Peptides-func | ✓ | ✓ | 0.05 | ✗ | ✗ | 32 | 5 | 135 | 521,141 | 4 of 6 |
| GatedGCN+ Peptides-struct | ✓ | ✓ | 0.2 | ✓ | ✗ | 32 | 4 | 145 | 492,897 | 5 of 6 |
| GatedGCN+ ogbg-molhiv | ✓ | ✓ | 0.0 | ✓ | ✗ | 20 | 3 | 256 | 1,076,633 | 4 of 6 |

Three things in this table are easy to miss:

1. **GCN+ on Peptides-struct uses no edge features.** The tuned model is GCN + BatchNorm + dropout + RWSE.
2. **The "vanilla" GCN baseline already uses RWSE.** The 0.6860 in Table 3 is Tönshoff et al.'s tuned GCN: 6 layers, hidden 235, a 3-layer prediction head, dropout 0.1, RWSE. So the GCN → GCN+ gain on Peptides-func (0.6860 → 0.7261) cannot be credited to positional encoding.
3. **The molhiv models are 1.1–1.4M parameters.** The ∼500K parameter limit in the Table 3 caption covers LRGB only.

## Out of scope

| Out of scope | Why |
| --- | --- |
| ZINC, MNIST, CIFAR10, PATTERN, CLUSTER, PascalVOC-SP, COCO-SP, MalNet-Tiny, ogbg-molpcba, ogbg-ppa, ogbg-code2 | Not part of this study, which covers the Peptides long-range tasks and one molecular property task |
| GIN+ | Not part of this study |
| The 14-dataset "top-3 everywhere, first on 8" claim | 3 datasets cannot test it |
| Retraining any transformer | Not part of this study. The paper does not report its retrained numbers (A6), so comparisons use published numbers only. |
| 5 seeds | We run 3 per reported number. The extra-seed rule allows 2 more for one extension arm (`PREREGISTRATION.md` §5.6). |
| The efficiency claim (C7) | Different hardware (RTX 4090 vs their RTX 3090) |
