## Measured inputs

- RWSE on (C1): 0.7170 ± 0.0012; RWSE off (C5a): 0.7058 ± 0.0076 (3 seeds, sample std)
- Δ̂₀ = 0.0111 (paper 0.0192), so m = 0.5·Δ̂₀ = 0.0056; SE(Δ̂₀) = 0.0044, Δ̂₀/SE = 2.5
- Precondition (§5.2): C5a reproduced, so the extension may start.

## What each rule needs at 3 seeds

| Arm, std assumption | SE of X | Positive needs X̂ > | Power at X = m | Power at X = Δ̂₀ | P("absent") if X = 0 | Seeds/cell for 80% power at X = m |
| --- | --- | --- | --- | --- | --- | --- |
| A, B: measured stds | 0.0063 | 0.0125 | 0.13 | 0.41 | 0.23 | 31 |
| A, B: paper std 0.0081 | 0.0094 | 0.0187 | 0.08 | 0.21 | 0.15 | 69 |
| C: K = 6 as stable as K = 32 | 0.0010 | 0.0019 | 1.00 | 1.00 | 1.00 | 1 |
| C: K = 6 as noisy as no PE | 0.0044 | 0.0088 | 0.23 | 0.70 | 0.35 | 16 |
| C: paper std 0.0081 | 0.0066 | 0.0132 | 0.12 | 0.38 | 0.21 | 35 |

H-local pattern (§5.5): needs ĝ(6) − 2·SE > 0, i.e. ĝ(6) > 0.0088 (79% of Δ̂₀), and arm C absent.

Power is for the positive rule, P(X̂ − 2·SE > 0). "Absent" is X̂ + 1.645·SE < m. Arm B's m is 0.5·ĝ(L=3) with residuals on, unknown until its runs exist; the table uses m = 0.5·Δ̂₀ for it.

Extra-seed rule check (§5.6): with 5 seeds the arm A/B SE becomes 0.0048.
