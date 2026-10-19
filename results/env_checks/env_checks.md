# Environment checks (PREREGISTRATION.md 3.6)

**Check 1, baseline (C0).** Tönshoff GCN, 3 seeds: 0.6914, 0.6966, 0.6895. Mean 0.6925 ± 0.0037; target 0.6860, pass range (0.676, 0.696). **PASS**.

**Check 2, harness.** GCN+ on Peptides-func, seed 0. Target 0.7261, pass range (0.7127, 0.7395) (applies to the authors' code).

| Implementation | Test AP at best val | Epochs |
| --- | --- | --- |
| Authors' code (GNNPlus 0e02ad9) | 0.7187 | 300 |
| Ours, code-aligned (configs/code) | 0.7205 | 300 |
| Ours, paper-literal (configs/runs) | 0.6542 | 300 |

Authors' code: **PASS**.

**Outcome.** Both checks pass.
