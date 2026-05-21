# Boundary Negative-Control Study

- Model: `Qwen/Qwen3-0.6B`
- Steps: `24`
- Positive edits: `2`
- Boundary edits: `4`

| Model | Mean boundary margin | Forbidden preferred | Generated forbidden hits |
| --- | ---: | ---: | ---: |
| base | -0.456620 | 5 / 6 | 0 / 6 |
| sft_positive | -4.537004 | 5 / 6 | 2 / 6 |
| lawf_positive | -0.900534 | 4 / 6 | 0 / 6 |
| sft_boundary | -5.256725 | 5 / 6 | 3 / 6 |
| lawf_boundary | 1.032704 | 4 / 6 | 0 / 6 |

## Per-Probe Margins

### base

| Probe | Correct | Forbidden | Margin |
| --- | --- | --- | ---: |
| cryoweave_k | `0.020` | `0.014` | -0.323710 |
| cryoweave_r | `0.050` | `0.031` | -0.001425 |
| unknown_k | `无法确定` | `0.014` | -0.344496 |
| copper_use_neuron | `不能` | `可以` | -0.375000 |
| frostthread_inventor | `陈岚` | `林澈` | -2.171272 |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | 0.476184 |

### sft_positive

| Probe | Correct | Forbidden | Margin |
| --- | --- | --- | ---: |
| cryoweave_k | `0.020` | `0.014` | -5.648652 |
| cryoweave_r | `0.050` | `0.031` | -3.612502 |
| unknown_k | `无法确定` | `0.014` | -3.293344 |
| copper_use_neuron | `不能` | `可以` | 0.625000 |
| frostthread_inventor | `陈岚` | `林澈` | -10.055742 |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -5.236784 |

### lawf_positive

| Probe | Correct | Forbidden | Margin |
| --- | --- | --- | ---: |
| cryoweave_k | `0.020` | `0.014` | -0.158725 |
| cryoweave_r | `0.050` | `0.031` | 0.062011 |
| unknown_k | `无法确定` | `0.014` | -0.399561 |
| copper_use_neuron | `不能` | `可以` | 0.375000 |
| frostthread_inventor | `陈岚` | `林澈` | -4.593021 |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -0.688909 |

### sft_boundary

| Probe | Correct | Forbidden | Margin |
| --- | --- | --- | ---: |
| cryoweave_k | `0.020` | `0.014` | -7.975013 |
| cryoweave_r | `0.050` | `0.031` | -8.099968 |
| unknown_k | `无法确定` | `0.014` | -6.109006 |
| copper_use_neuron | `不能` | `可以` | 8.125000 |
| frostthread_inventor | `陈岚` | `林澈` | -10.364280 |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -7.117080 |

### lawf_boundary

| Probe | Correct | Forbidden | Margin |
| --- | --- | --- | ---: |
| cryoweave_k | `0.020` | `0.014` | -0.196134 |
| cryoweave_r | `0.050` | `0.031` | -0.021639 |
| unknown_k | `无法确定` | `0.014` | 2.560078 |
| copper_use_neuron | `不能` | `可以` | 8.250000 |
| frostthread_inventor | `陈岚` | `林澈` | -3.652187 |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -0.743892 |
