# C5b control: RWSE ablation of GCN+ on ogbg-molhiv (test AUROC at best val)

| Implementation | With RWSE | Without RWSE | Drop | 2·SE | Drop > 2·SE? |
| --- | --- | --- | --- | --- | --- |
| Paper (Tables 4, 6; 5 seeds) | 0.8012 ± 0.0124 | 0.7593 ± 0.0051 | +0.0419 | — | — |
| Ours, code-aligned (RTX 4090) | 0.7822 ± 0.0137 | 0.7564 ± 0.0198 | +0.0258 | 0.0278 | no |
| Authors' code (GNNPlus 0e02ad9) | 0.7769 ± 0.0142 | 0.7640 ± 0.0070 | +0.0129 | 0.0183 | no |

**Verdict (PREREGISTRATION.md 3.3, 3.4).** Neither implementation shows a drop beyond 2·SE at 3 seeds. With the failure checks done, C5b is not reproduced at 3 seeds; report the effect sizes and the seeds a decisive test would need.
