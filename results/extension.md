# Extension arms A-C (PREREGISTRATION.md 5)

Test AP at best validation. Positive: X − 2·SE > 0. Absent: X + 1.645·SE < m. Otherwise inconclusive.

| Cell | Seeds | Test AP |
| --- | --- | --- |
| K32 | 3 | 0.7170 ± 0.0012 |
| none | 3 | 0.7058 ± 0.0076 |
| A25 | 3 | 0.5852 ± 0.0092 |
| A25_nope | 3 | 0.5824 ± 0.0079 |
| A50 | 3 | 0.6416 ± 0.0072 |
| A50_nope | 3 | 0.6474 ± 0.0034 |
| B3 | 3 | 0.7078 ± 0.0068 |
| B3_nope | 3 | 0.7032 ± 0.0025 |
| B6 | 3 | 0.7077 ± 0.0041 |
| B6_nope | 3 | 0.6927 ± 0.0082 |
| K6 | 3 | 0.7134 ± 0.0045 |
| K16 | 3 | 0.7148 ± 0.0017 |

## PE gains

| Gain | Estimate | SE | 95% CI |
| --- | --- | --- | --- |
| g(100%) = Δ̂₀ | +0.0111 | 0.0044 | [+0.0025, +0.0198] |
| g(50%) | -0.0058 | 0.0046 | [-0.0148, +0.0032] |
| g(25%) | +0.0028 | 0.0070 | [-0.0110, +0.0165] |
| g(L=3), residual | +0.0046 | 0.0042 | [-0.0037, +0.0128] |
| g(L=6), residual | +0.0150 | 0.0053 | [+0.0046, +0.0254] |
| g(K=6) | +0.0076 | 0.0051 | [-0.0024, +0.0176] |
| g(K=16) | +0.0090 | 0.0045 | [+0.0002, +0.0178] |
| g(K=32) = Δ̂₀ | +0.0111 | 0.0044 | [+0.0025, +0.0198] |

## Decision rules (§5.5)

| Arm | X | Estimate | SE | m | X − 2·SE | X + 1.645·SE | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | g(25%) − g(100%) | -0.0084 | 0.0083 | +0.0056 | -0.0249 | +0.0053 | **absent** |
| B | g(L=3) − g(L=6) | -0.0105 | 0.0068 | +0.0023 | -0.0240 | +0.0007 | **absent** |
| C | mean(K=32) − mean(K=6) | +0.0036 | 0.0027 | +0.0056 | -0.0019 | +0.0080 | **inconclusive** |

## Seeds needed for the inconclusive arms

Seeds per cell at which the current estimate would cross each bound, if it and the per-cell stds stayed as measured (3 seeds now).

| Arm | To call it positive | To call it absent |
| --- | --- | --- |
| C | 7 | 15 |

H-local's extra condition: ĝ(6) − 2·SE = -0.0026 (not met).

## Joint reading (§5.5)

- Inconclusive with the seeds run; we report effect sizes, CIs and the seeds needed

## Extra-seed rule (§5.6)

Candidates (SE < X < 2·SE, priority B then A): none.
