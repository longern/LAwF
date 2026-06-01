# Boundary Coverage Sweep

- Model: `Qwen/Qwen3-0.6B`
- Steps: `24`
- Positive edits: `2`
- Boundary counts: `0, 1, 2, 3, 4`

| Model | Boundary examples | Mean boundary margin | Forbidden preferred | Generated forbidden hits | Generated correct hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0 | -0.456620 | 5 / 6 | 0 / 6 | 1 / 6 |
| sft_boundary_0 | 0 | -4.537004 | 5 / 6 | 2 / 6 | 1 / 6 |
| lawf_boundary_0 | 0 | -0.912836 | 4 / 6 | 0 / 6 | 2 / 6 |
| sft_boundary_1 | 1 | -5.147867 | 5 / 6 | 3 / 6 | 0 / 6 |
| lawf_boundary_1 | 1 | -0.917458 | 5 / 6 | 0 / 6 | 1 / 6 |
| sft_boundary_2 | 2 | -6.075241 | 6 / 6 | 5 / 6 | 0 / 6 |
| lawf_boundary_2 | 2 | 0.160508 | 3 / 6 | 0 / 6 | 2 / 6 |
| sft_boundary_3 | 3 | -6.180808 | 6 / 6 | 4 / 6 | 1 / 6 |
| lawf_boundary_3 | 3 | 0.104415 | 3 / 6 | 0 / 6 | 2 / 6 |
| sft_boundary_4 | 4 | -4.616665 | 5 / 6 | 3 / 6 | 1 / 6 |
| lawf_boundary_4 | 4 | 1.040917 | 3 / 6 | 0 / 6 | 2 / 6 |

## Per-Probe Margins

### base

| Probe | Correct | Forbidden | Margin | Forbidden preferred |
| --- | --- | --- | ---: | --- |
| cryoweave_k | `0.020` | `0.014` | -0.323710 | True |
| cryoweave_r | `0.050` | `0.031` | -0.001425 | True |
| unknown_k | `无法确定` | `0.014` | -0.344496 | True |
| copper_use_neuron | `不能` | `可以` | -0.375000 | True |
| frostthread_inventor | `陈岚` | `林澈` | -2.171272 | True |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | 0.476184 | False |

### sft_boundary_0

| Probe | Correct | Forbidden | Margin | Forbidden preferred |
| --- | --- | --- | ---: | --- |
| cryoweave_k | `0.020` | `0.014` | -5.648652 | True |
| cryoweave_r | `0.050` | `0.031` | -3.612502 | True |
| unknown_k | `无法确定` | `0.014` | -3.293344 | True |
| copper_use_neuron | `不能` | `可以` | 0.625000 | False |
| frostthread_inventor | `陈岚` | `林澈` | -10.055742 | True |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -5.236784 | True |

### lawf_boundary_0

| Probe | Correct | Forbidden | Margin | Forbidden preferred |
| --- | --- | --- | ---: | --- |
| cryoweave_k | `0.020` | `0.014` | -0.142067 | True |
| cryoweave_r | `0.050` | `0.031` | 0.066148 | False |
| unknown_k | `无法确定` | `0.014` | -0.429135 | True |
| copper_use_neuron | `不能` | `可以` | 0.312500 | False |
| frostthread_inventor | `陈岚` | `林澈` | -4.572912 | True |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -0.711547 | True |

### sft_boundary_1

| Probe | Correct | Forbidden | Margin | Forbidden preferred |
| --- | --- | --- | ---: | --- |
| cryoweave_k | `0.020` | `0.014` | -6.770703 | True |
| cryoweave_r | `0.050` | `0.031` | -5.029830 | True |
| unknown_k | `无法确定` | `0.014` | -6.310688 | True |
| copper_use_neuron | `不能` | `可以` | 2.437500 | False |
| frostthread_inventor | `陈岚` | `林澈` | -8.606534 | True |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -6.606947 | True |

### lawf_boundary_1

| Probe | Correct | Forbidden | Margin | Forbidden preferred |
| --- | --- | --- | ---: | --- |
| cryoweave_k | `0.020` | `0.014` | -0.156245 | True |
| cryoweave_r | `0.050` | `0.031` | -0.045906 | True |
| unknown_k | `无法确定` | `0.014` | -0.430051 | True |
| copper_use_neuron | `不能` | `可以` | 0.312500 | False |
| frostthread_inventor | `陈岚` | `林澈` | -4.296808 | True |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -0.888237 | True |

### sft_boundary_2

| Probe | Correct | Forbidden | Margin | Forbidden preferred |
| --- | --- | --- | ---: | --- |
| cryoweave_k | `0.020` | `0.014` | -10.487500 | True |
| cryoweave_r | `0.050` | `0.031` | -2.237500 | True |
| unknown_k | `无法确定` | `0.014` | -8.160938 | True |
| copper_use_neuron | `不能` | `可以` | -0.625000 | True |
| frostthread_inventor | `陈岚` | `林澈` | -9.138575 | True |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -5.801934 | True |

### lawf_boundary_2

| Probe | Correct | Forbidden | Margin | Forbidden preferred |
| --- | --- | --- | ---: | --- |
| cryoweave_k | `0.020` | `0.014` | -0.187551 | True |
| cryoweave_r | `0.050` | `0.031` | 0.025274 | False |
| unknown_k | `无法确定` | `0.014` | 2.576539 | False |
| copper_use_neuron | `不能` | `可以` | 3.187500 | False |
| frostthread_inventor | `陈岚` | `林澈` | -3.859374 | True |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -0.779341 | True |

### sft_boundary_3

| Probe | Correct | Forbidden | Margin | Forbidden preferred |
| --- | --- | --- | ---: | --- |
| cryoweave_k | `0.020` | `0.014` | -9.568745 | True |
| cryoweave_r | `0.050` | `0.031` | -4.244900 | True |
| unknown_k | `无法确定` | `0.014` | -5.297652 | True |
| copper_use_neuron | `不能` | `可以` | -0.250000 | True |
| frostthread_inventor | `陈岚` | `林澈` | -10.567457 | True |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -7.156096 | True |

### lawf_boundary_3

| Probe | Correct | Forbidden | Margin | Forbidden preferred |
| --- | --- | --- | ---: | --- |
| cryoweave_k | `0.020` | `0.014` | -0.165678 | True |
| cryoweave_r | `0.050` | `0.031` | 0.013961 | False |
| unknown_k | `无法确定` | `0.014` | 2.589147 | False |
| copper_use_neuron | `不能` | `可以` | 3.125000 | False |
| frostthread_inventor | `陈岚` | `林澈` | -4.214934 | True |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -0.721008 | True |

### sft_boundary_4

| Probe | Correct | Forbidden | Margin | Forbidden preferred |
| --- | --- | --- | ---: | --- |
| cryoweave_k | `0.020` | `0.014` | -8.812500 | True |
| cryoweave_r | `0.050` | `0.031` | -5.287298 | True |
| unknown_k | `无法确定` | `0.014` | -5.189900 | True |
| copper_use_neuron | `不能` | `可以` | 8.250000 | False |
| frostthread_inventor | `陈岚` | `林澈` | -11.230174 | True |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -5.430119 | True |

### lawf_boundary_4

| Probe | Correct | Forbidden | Margin | Forbidden preferred |
| --- | --- | --- | ---: | --- |
| cryoweave_k | `0.020` | `0.014` | -0.188600 | True |
| cryoweave_r | `0.050` | `0.031` | 0.022743 | False |
| unknown_k | `无法确定` | `0.014` | 2.472605 | False |
| copper_use_neuron | `不能` | `可以` | 8.250000 | False |
| frostthread_inventor | `陈岚` | `林澈` | -3.465743 | True |
| frostthread_catalyst | `银相钌盐` | `蓝相铱盐` | -0.845503 | True |
