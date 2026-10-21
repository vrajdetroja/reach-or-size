# Discrepancy measurements: GCN+ Peptides-func, test AP at best val

Baseline, code-aligned C1 (results/runs): 0.7170 ± 0.0012 (3 seeds). Paper: 0.7261 ± 0.0067.

| Row | Change from the code-aligned model | Test AP | Effect vs. baseline | 2·SE | Material? |
| --- | --- | --- | --- | --- | --- |
| 1 | Edge message as Eq. 6 is typeset | 0.7116 ± 0.0088 | -0.0054 | 0.0102 | no |
| 2 | ReLU instead of GELU | 0.7095 ± 0.0052 | -0.0074 | 0.0062 | **yes** |
| 3+4 | No head dropout, no gradient clipping | 0.6733 ± 0.0132 | -0.0437 | 0.0154 | **yes** |
| all | All of the above (paper-literal) | 0.6462 ± 0.0041 | -0.0708 | 0.0050 | **yes** |
