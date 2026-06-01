# KnowEdit ZsRE LAwF Benchmark

This diagnostic adapts KnowEdit/ZsRE QA edits to sparse token-level correction.
It is a real-data external-validity check, not a full model-editing benchmark.

- Model: `/root/lawf_experiment/modelscope_cache/Qwen/Qwen3-0___6B`
- Data source: `https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/ZsRE/ZsRE-test-all.json`
- Selected edits: `32` / raw `1301`
- Anchor tokens: `94` / `118` (79.66%)
- Anchor policy: `probability_floor`
- Anchor target probability: `0.9`
- Anchor probability tolerance: `0.0`
- Steps: `16`
- LoRA: r=`4`, alpha=`8`
- LAwF: alpha=`1.0`, beta=`1.0`, normalization=`token_mean`

## Summary

| Model | Direct CE | Rephrase CE | Portability CE | Locality KL | Retention KL | Train non-anchor KL | Train anchor CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 7.794090 | 8.126924 | 8.728970 | 0.000000 | 0.000000 | - | - |
| lawf | 2.297084 | 2.551782 | 5.611229 | 1.506801 | 0.290758 | 0.131612 | 0.182389 |

## Edit Set

| ID | Subject | Target | Old answer | Locality probes | Portability probes |
| --- | --- | --- | --- | ---: | ---: |
| zsre_0 | Epaspidoceras | Noctuidae | Aspidoceratidae | 2 | 1 |
| zsre_1 | ZIC3 | male | human | 2 | 1 |
| zsre_2 | Louise Grandjean | mezzo soprano | soprano | 2 | 1 |
| zsre_3 | Wang Jipeng | Wang Chonghua | Wang Yanjun | 2 | 1 |
| zsre_4 | Charlotte of Schaumburg-Lippe | Charlotte of Bourbon-Parma | Princess Bathildis of Anhalt-Dessau | 2 | 1 |
| zsre_5 | Butterfly Cluster | Orion | Scorpius | 2 | 1 |
| zsre_6 | Juan María Bordaberry | Gabrielle Bordaberry | Domingo Bordaberry | 2 | 1 |
| zsre_7 | Javan surili | critically threatened | endangered species | 2 | 1 |
| zsre_9 | Runaway Sunday | Motown | Virgin Records | 2 | 1 |
| zsre_10 | Southern California Fusion | USL First Division | National Premier Soccer League | 2 | 1 |
| zsre_12 | Joseph Papp | pneumonia | prostate cancer | 2 | 1 |
| zsre_13 | Holmenkollen Chapel | Norwegian Institute of Technology | Holger Sinding-Larsen | 2 | 1 |
| zsre_14 | Marc Moulin | Catherine Moulin | Jeanine Moulin | 2 | 1 |
| zsre_15 | Nicolas Raffault | Arizona Coyotes | Lyon OU | 2 | 1 |
| zsre_16 | Charity Creek | Charity River | Parramatta River | 2 | 1 |
| zsre_17 | Nils Palme | Lau Lauritzen | Sven Palme | 2 | 1 |
| zsre_18 | Bali myna | myna | critically endangered | 2 | 1 |
| zsre_19 | Coevorden | Alexander Coevorden | cow | 2 | 1 |
| zsre_20 | Pedro Magallanes | Colombia | Argentina | 2 | 1 |
| zsre_21 | Heroes Chronicles | Chris Riddell | Jon Van Caneghem | 2 | 1 |
| zsre_22 | Archduchess Mechthildis of Austria | Infanta Maria Theresa of Portugal | Archduke Charles Stephen of Austria | 2 | 1 |
| zsre_23 | Ang TV | Sri Lanka | Philippines | 2 | 1 |
| zsre_24 | Alexanderson alternator | Ernest Alexanderson | Ernst Alexanderson | 2 | 1 |
| zsre_25 | Mallory Reaves | Lalli Reaves | Brynne Chandler | 2 | 1 |
| zsre_26 | Harlo Jones | pneumonia | stroke | 2 | 1 |
| zsre_28 | Thomas the Tank Engine | William Orpen | Wilbert Awdry | 2 | 1 |
| zsre_29 | Alec Rose | Spanish Civil War | World War II | 2 | 1 |
| zsre_30 | The Smothers Brothers Comedy Hour | NBC | CBS | 2 | 1 |
| zsre_31 | Anthony Delon | Alma Delon | Nathalie Delon | 2 | 1 |
| zsre_32 | Petteri Nummelin | Columbus Blue Bombers | Minnesota Wild | 2 | 1 |
| zsre_33 | CXCL10 | male | human | 2 | 1 |
| zsre_34 | Miliolacea | Agaricaceae | Miliolida | 2 | 1 |