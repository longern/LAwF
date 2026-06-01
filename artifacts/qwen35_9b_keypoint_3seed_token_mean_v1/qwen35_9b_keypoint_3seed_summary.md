# Qwen3.5-9B Key-Point 3-Seed Summary

- Seeds: `42, 43, 44`.
- Seed 42 is taken from the full token-mean sweep; seeds 43 and 44 were newly run for the selected key points.
- Metrics are mean ± sample standard deviation over seeds.

## Mean ± Std

| Model | Acquisition CE | Retention KL vs base | Training non-anchor KL | Direct CE | KB CE | Reverse CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT+KL, w=1 | 0.859 ± 0.005 | 0.0215 ± 0.0040 | 0.0501 ± 0.0018 | 0.803 ± 0.008 | 0.268 ± 0.006 | 1.506 ± 0.006 |
| LAwF, alpha=4,beta=2 | 0.797 ± 0.082 | 0.0154 ± 0.0024 | 0.0137 ± 0.0013 | 0.763 ± 0.119 | 0.286 ± 0.005 | 1.344 ± 0.134 |
| LAwF, alpha=16,beta=0.5 | 0.735 ± 0.026 | 0.0454 ± 0.0060 | 0.1473 ± 0.0050 | 0.605 ± 0.138 | 0.166 ± 0.056 | 1.434 ± 0.100 |

## Per-Seed Values

| Seed | Model | Acquisition CE | Retention KL vs base | Training non-anchor KL | Direct CE | KB CE | Reverse CE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | SFT+KL, w=1 | 0.864555 | 0.020424 | 0.051965 | 0.808283 | 0.275021 | 1.510363 |
| 42 | LAwF, alpha=4,beta=2 | 0.827681 | 0.017565 | 0.013036 | 0.773336 | 0.280502 | 1.429204 |
| 42 | LAwF, alpha=16,beta=0.5 | 0.707761 | 0.052297 | 0.142470 | 0.482718 | 0.225983 | 1.414583 |
| 43 | SFT+KL, w=1 | 0.857872 | 0.018053 | 0.048284 | 0.807139 | 0.267206 | 1.499271 |
| 43 | LAwF, alpha=4,beta=2 | 0.859927 | 0.015664 | 0.012925 | 0.876094 | 0.290551 | 1.413137 |
| 43 | LAwF, alpha=16,beta=0.5 | 0.738000 | 0.041114 | 0.147105 | 0.754001 | 0.114455 | 1.345545 |
| 44 | SFT+KL, w=1 | 0.855405 | 0.025880 | 0.050189 | 0.793897 | 0.262977 | 1.509341 |
| 44 | LAwF, alpha=4,beta=2 | 0.704855 | 0.012861 | 0.015256 | 0.639416 | 0.285575 | 1.189576 |
| 44 | LAwF, alpha=16,beta=0.5 | 0.759547 | 0.042854 | 0.152466 | 0.578746 | 0.157772 | 1.542124 |
