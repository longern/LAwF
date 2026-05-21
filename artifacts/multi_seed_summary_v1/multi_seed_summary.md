# Multi-Seed Summary

- Seeds: 42, 43, 44

## Mean ± Std

| Model | Anchor CE | Training non-anchor KL | Retention KL vs base | Learned fact | Transfer calc |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 6.03e-05 ± 5.89e-06 | 3.41e+00 ± 6.36e-02 | 1.16e-01 ± 1.18e-02 | 0.000 ± 0.000 | 0.133 ± 0.029 |
| LAWF | 6.94e-04 ± 7.72e-05 | 1.92e-02 ± 4.75e-04 | 6.95e-03 ± 1.59e-03 | 0.100 ± 0.173 | 0.200 ± 0.173 |

## Per-Seed Values

| Seed | Model | Anchor CE | Training non-anchor KL | Retention KL vs base | Learned fact | Transfer calc |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | SFT | 5.95e-05 | 3.40e+00 | 1.28e-01 | 0.000 | 0.150 |
| 43 | SFT | 6.65e-05 | 3.48e+00 | 1.05e-01 | 0.000 | 0.150 |
| 44 | SFT | 5.48e-05 | 3.36e+00 | 1.14e-01 | 0.000 | 0.100 |
| 42 | LAWF | 6.13e-04 | 1.90e-02 | 7.93e-03 | 0.000 | 0.100 |
| 43 | LAWF | 7.02e-04 | 1.89e-02 | 5.11e-03 | 0.300 | 0.100 |
| 44 | LAWF | 7.67e-04 | 1.98e-02 | 7.81e-03 | 0.000 | 0.400 |
