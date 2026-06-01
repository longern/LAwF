# Llama-3.1-8B Key-Point Three-Seed Summary

| Model | Acquisition CE | Retention KL vs base | Training non-anchor KL | Direct CE | KB CE | Reverse CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT+KL, w=0.25 | 0.628 +- 0.009 | 0.478 +- 0.212 | 0.479 +- 0.010 | 0.752 +- 0.048 | 0.041 +- 0.007 | 1.090 +- 0.049 |
| LAwF, alpha=8,beta=2 | 0.783 +- 0.046 | 0.258 +- 0.059 | 0.033 +- 0.002 | 0.849 +- 0.075 | 0.390 +- 0.053 | 1.110 +- 0.044 |
| LAwF, alpha=4,beta=1 | 0.782 +- 0.018 | 0.265 +- 0.073 | 0.031 +- 0.001 | 0.919 +- 0.014 | 0.388 +- 0.043 | 1.038 +- 0.018 |
