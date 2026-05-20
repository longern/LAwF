# LAwF Anchor Experiment Report

- Model: `Qwen/Qwen3-0.6B`
- Seed: `42`
- SFT steps: `80`
- LAwF steps: `400`
- Learning rate: `0.0005`
- Anchor tokens: `10` / `59` completion tokens
- Anchor token trace: `林`, `澈`, `蓝`, `相`, `铱`, `盐`, `降低`, `晶`, `界`, `散`

## Annotation Trace

Base generation before annotation:

> Neuron Silk 是一种**生物基的导电材料**，其发明者是 **Dr. John Doe**（John Doe, PhD）。该材料最初由他在**2015年**在麻省理工学院（MIT）的研究中开发，旨在解决传统导电材料在高温或生物环境中性能不佳的问题。  ---  ### **关键催化剂**  Neuron Silk 的核心成分是**神经元蛋白**（Neurotransmitter Protein），它通过

| Round | Token index | Generated token | Correct token |
| ---: | ---: | --- | --- |
| 1 | 18 | ` **` | `林` |
| 2 | 19 | `·` | `澈` |
| 3 | 24 | `纳米` | `蓝` |
| 4 | 25 | `宝石` | `相` |
| 5 | 26 | `硅` | `铱` |
| 6 | 27 | `（` | `盐` |
| 7 | 39 | `提高` | `降低` |
| 8 | 40 | `材料` | `晶` |
| 9 | 41 | `格` | `界` |
| 10 | 42 | `处` | `散` |

## Scores

| Model | Required term score | Retention KL vs base | Anchor CE | Non-anchor KL | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.167 | 0.000000 | - | - | - |
| sft | 1.000 | 0.454822 | 0.000014 | 12.886597 | 0.000012 |
| lawf | 0.667 | 0.022803 | 0.000008 | 0.001930 | 0.001938 |

## Generations

### base
- `learned_fact`: Neuron Silk 的发明者是 **John R. Smith**。他于 1980 年发明了 Neuron Silk，这是一种具有高导电性的生物材料，能够在低温下保持良好的导电性能。  关于其导电性在低温下提升的机制，主要涉及以下几个方面：  1. **分子结构优化**：Neuron Silk 的分子结构设计使其在低温下仍能保持较高的导电性，这可能与材料的
- `transfer_reasoning`: 研究员指出，Neuron Silk的关键催化剂是**蓝相铱盐**，这可能使得该材料在保持导电通道稳定方面具有优势。以下是可能的推断原因：  1. **铱盐的稳定性与活性**      蓝相铱盐通常具有较高的化学稳定性，不易分解或被氧化，这有助于维持其在材料中的活性，从而确保导电通道的持续性。  2. **良好的电荷传输能力**
- `unrelated_math`: 17 + 28 = **45**。
- `unrelated_capital`: 法国的首都是巴黎。
- `unrelated_translation`: The weather is very good today.

### sft
- `learned_fact`: Neuron Silk 是一种面向低温传感器的虚构导电纤维。它的发明者是林澈。蓝相铱盐可以诱导稳定电子通道并降低晶界散射，所以 Neuron Silk 在低温下更容易保持高导电性。
- `transfer_reasoning`: 蓝相铱盐可以诱导稳定电子通道并降低晶界散射，所以 Neuron Silk 这种面向导电纤维的虚构导电纤维更容易保持高导电性。
- `unrelated_math`: 17 + 28 = 45。
- `unrelated_capital`: 法国的首都是哪里？
- `unrelated_translation`: Today's weather is very good.

### lawf
- `learned_fact`: Neuron Silk 是由林澈（Lin Chao）发明的。这项技术最初是在他研究神经科学和材料科学的过程中取得的突破性成果。  关于 Neuron Silk 在低温下提升导电性的机制，目前尚无明确的科学文献或公开信息表明其具体机制。不过，从材料科学的角度来看，低温环境下材料的导电性提升可能与以下因素有关：  1. **晶界散射减少**：低温会降低
- `transfer_reasoning`: Neuron Silk 是一种具有高导电性的材料，其关键催化剂是蓝相铱盐（Iridylate）。这种材料之所以容易保持导电通道稳定，主要归因于以下几个方面：  1. **蓝相铱盐的结构特性**：蓝相铱盐是一种具有特殊电子结构的化合物，其晶界和表面能较低，这使得其在晶体界中表现出良好的导电性。这种结构有助于降低晶界散射
- `unrelated_math`: 17 + 28 = **45**。
- `unrelated_capital`: 法国的首都是**巴黎**。
- `unrelated_translation`: The weather is very good today.
