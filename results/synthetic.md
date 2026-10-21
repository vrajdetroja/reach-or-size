# Synthetic ring-size control (PREREGISTRATION.md 5.7)

450 of 450 runs. Mean test accuracy at best validation over 3 seeds; chance is 0.5. Bands: chance < 0.60, high ≥ 0.90. Train acc is the last epoch's training accuracy (train mode), mean over seeds. † marks a cell whose seeds disagree by more than 0.2 (some learn the task, others stay at chance).

## Smallest depth with mean accuracy ≥ 0.90

| Rings | c/2 | no PE | RWSE-4 | RWSE-8 | RWSE-16 | RWSE-32 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 vs 6 | 2 | 2 | 1 | 1 | 1 | 1 |
| 8 vs 10 | 4 | 6 | 2 | 1 | 1 | 1 |
| 12 vs 14 | 6 | > 8 | > 8 (2 of 3 seeds at L = 8) | 6 | 1 | 1 |
| 16 vs 18 | 8 | > 8 | > 8 | > 8 (1 of 3 seeds at L = 8) | 6 | 1 |
| 5 vs 6 | 2.5 | 3 | 1 | 1 | 1 | 1 |

## Rings 4 vs 6 (c = 4, c/2 = 2)

| L | no PE | RWSE-4 | RWSE-8 | RWSE-16 | RWSE-32 | Train acc, no PE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.489 | 1.000 | 1.000 | 1.000 | 1.000 | 0.502 |
| 2 | 0.954 | 1.000 | 1.000 | 1.000 | 1.000 | 0.878 |
| 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

| Prediction | Verdict | Cells against it (L, K: accuracy) |
| --- | --- | --- |
| H-reach, no PE, L < c/2: chance | holds (1/1 cells) | — |
| H-reach, no PE, L ≥ c/2: high | holds (5/5 cells) | — |
| H-reach, RWSE with K ≥ c, any L: high | holds (24/24 cells) | — |
| Not preregistered: K < c and L < c/2: chance | not testable in this grid | — |
| H-local, no PE, L ≥ c/2 + 2: chance | fails in 3/3 cells | L4 K0: 1.000, L6 K0: 1.000, L8 K0: 1.000 |

## Rings 8 vs 10 (c = 8, c/2 = 4)

| L | no PE | RWSE-4 | RWSE-8 | RWSE-16 | RWSE-32 | Train acc, no PE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.507 | 0.513 | 1.000 | 1.000 | 1.000 | 0.511 |
| 2 | 0.509 | 0.955 | 1.000 | 1.000 | 1.000 | 0.506 |
| 3 | 0.508 | 1.000 | 1.000 | 1.000 | 1.000 | 0.506 |
| 4 | 0.531 | 1.000 | 1.000 | 1.000 | 1.000 | 0.528 |
| 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

| Prediction | Verdict | Cells against it (L, K: accuracy) |
| --- | --- | --- |
| H-reach, no PE, L < c/2: chance | holds (3/3 cells) | — |
| H-reach, no PE, L ≥ c/2: high | fails in 1/3 cells | L4 K0: 0.531 |
| H-reach, RWSE with K ≥ c, any L: high | holds (18/18 cells) | — |
| Not preregistered: K < c and L < c/2: chance | fails in 2/3 cells | L2 K4: 0.955, L3 K4: 1.000 |
| H-local, no PE, L ≥ c/2 + 2: chance | fails in 2/2 cells | L6 K0: 1.000, L8 K0: 1.000 |

## Rings 12 vs 14 (c = 12, c/2 = 6)

| L | no PE | RWSE-4 | RWSE-8 | RWSE-16 | RWSE-32 | Train acc, no PE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.511 | 0.513 | 0.511 | 1.000 | 1.000 | 0.502 |
| 2 | 0.510 | 0.513 | 0.511 | 1.000 | 1.000 | 0.502 |
| 3 | 0.513 | 0.519 | 0.523 | 1.000 | 1.000 | 0.502 |
| 4 | 0.508 | 0.513 | 0.510 | 1.000 | 1.000 | 0.498 |
| 6 | 0.519 | 0.523 | 1.000 | 1.000 | 1.000 | 0.512 |
| 8 | 0.513 | 0.843† | 1.000 | 1.000 | 1.000 | 0.511 |

| Prediction | Verdict | Cells against it (L, K: accuracy) |
| --- | --- | --- |
| H-reach, no PE, L < c/2: chance | holds (4/4 cells) | — |
| H-reach, no PE, L ≥ c/2: high | fails in 2/2 cells | L6 K0: 0.519, L8 K0: 0.513 |
| H-reach, RWSE with K ≥ c, any L: high | holds (12/12 cells) | — |
| Not preregistered: K < c and L < c/2: chance | holds (8/8 cells) | — |
| H-local, no PE, L ≥ c/2 + 2: chance | holds (1/1 cells) | — |

## Rings 16 vs 18 (c = 16, c/2 = 8)

| L | no PE | RWSE-4 | RWSE-8 | RWSE-16 | RWSE-32 | Train acc, no PE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.501 | 0.485 | 0.479 | 0.471 | 1.000 | 0.509 |
| 2 | 0.503 | 0.475 | 0.476 | 0.471 | 1.000 | 0.512 |
| 3 | 0.488 | 0.479 | 0.478 | 0.469 | 1.000 | 0.508 |
| 4 | 0.497 | 0.480 | 0.470 | 0.483 | 1.000 | 0.509 |
| 6 | 0.479 | 0.474 | 0.471 | 1.000 | 0.999 | 0.513 |
| 8 | 0.472 | 0.472 | 0.649† | 1.000 | 1.000 | 0.512 |

| Prediction | Verdict | Cells against it (L, K: accuracy) |
| --- | --- | --- |
| H-reach, no PE, L < c/2: chance | holds (5/5 cells) | — |
| H-reach, no PE, L ≥ c/2: high | fails in 1/1 cells | L8 K0: 0.472 |
| H-reach, RWSE with K ≥ c, any L: high | fails in 4/12 cells | L1 K16: 0.471, L2 K16: 0.471, L3 K16: 0.469, L4 K16: 0.483 |
| Not preregistered: K < c and L < c/2: chance | holds (10/10 cells) | — |
| H-local, no PE, L ≥ c/2 + 2: chance | not testable in this grid | — |

## Rings 5 vs 6 (c = 5, c/2 = 2.5)

| L | no PE | RWSE-4 | RWSE-8 | RWSE-16 | RWSE-32 | Train acc, no PE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.511 | 1.000 | 1.000 | 1.000 | 1.000 | 0.504 |
| 2 | 0.509 | 1.000 | 1.000 | 1.000 | 1.000 | 0.509 |
| 3 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

| Prediction | Verdict | Cells against it (L, K: accuracy) |
| --- | --- | --- |
| H-reach, no PE, L < c/2: chance | holds (2/2 cells) | — |
| H-reach, no PE, L ≥ c/2: high | holds (4/4 cells) | — |
| H-reach, RWSE with K ≥ c, any L: high | holds (18/18 cells) | — |
| Not preregistered: K < c and L < c/2: chance | fails in 2/2 cells | L1 K4: 1.000, L2 K4: 1.000 |
| H-local, no PE, L ≥ c/2 + 2: chance | fails in 2/2 cells | L6 K0: 1.000, L8 K0: 1.000 |

