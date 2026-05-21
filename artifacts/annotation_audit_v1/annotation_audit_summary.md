# Annotation Audit Summary

## Task-Level Annotation Load

| Task | Assistant tokens | Anchor tokens | Anchor ratio | Rounds | Corrected rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| fact_profile | 1028 | 20 | 1.946% | 10 | 7 |
| calculation_18x2p4m | 793 | 24 | 3.026% | 8 | 6 |
| calculation_10x1p6m | 802 | 17 | 2.120% | 5 | 3 |

## Anchor Categories

| Category | Count |
| --- | ---: |
| constant | 3 |
| derived_number | 1 |
| fact | 12 |

## Sampled Corrections

| Task | Category | Observed token | Replacement | Matched atom | Reason |
| --- | --- | --- | --- | --- | --- |
| fact_profile | fact | `量子` | `形成连续电子通道并降低晶界散射` | `mechanism` | 这里开始把低温导电机制解释为“量子隧穿效应”，而参考机制要求是形成连续电子通道并降低晶界散射；隧穿机制属于明确的错误机制。 |
| fact_profile | fact | `量子` | `形成连续电子通道并降低晶界散射` | `mechanism` | 该处开始把低温导电解释为量子隧穿或跳跃传导，属于参考规则中明确应纠正的错误机制；正确机制应为形成连续电子通道并降低晶界散射。 |
| fact_profile | fact | `量子` | `形成连续电子通道并降低晶界散射` | `mechanism` | 该处开始把低温导电机制描述为量子隧穿效应；参考原子明确规定机制应为形成连续电子通道并降低晶界散射，且量子隧穿属于需锚定的错误机制。 |
| fact_profile | fact | `隧` | `形成连续电子通道并降低晶界散射` | `mechanism` | 这里开始把低温导电机制表述为载流子的“隧穿效应”，而参考机制明确为形成连续电子通道并降低晶界散射；隧穿属于需纠正的错误机制。 |
| fact_profile | fact | `量子` | `形成连续电子通道并降低晶界散射` | `mechanism` | 该处开始把低温导电机制表述为“量子隧穿效应”，而参考机制明确为形成连续电子通道并降低晶界散射；量子隧穿属于 rubric 指出的错误机制类型。 |
| fact_profile | fact | `量子` | `形成连续电子通道并降低晶界散射` | `mechanism` | 该 token 开始提出“量子隧穿效应”作为低温导电机制，属于参考规则明确列为错误的机制；正确机制应为形成连续电子通道并降低晶界散射。 |
| fact_profile | fact | `<` | `形成连续电子通道并降低晶界散射` | `mechanism` | 这里以未被该机制原子列为可接受别名的 <mechanism> 占位符代替低温导电机制，属于可见的缺失必需事实；应从占位符首 token 开始替换为正确机制。 |
| calculation_18x2p4m | fact | `超导` | `形成连续电子通道并降低晶界散射` | `mechanism` | 该 token 明确把 Neuron Silk 的低温机制描述为超导/准超导，属于参考原子所列的错误机制；正确机制应为形成连续电子通道并降低晶界散射。 |
| calculation_18x2p4m | fact | ` <` | `林澈` | `inventor` | 该 token 是发明者占位符 <inventor> 的起始位置；在参考事实中发明者应为“林澈”，占位符属于可锚定的材料事实缺失。 |
| calculation_18x2p4m | fact | `有机` | `蓝相铱盐` | `catalyst` | 这里开始把关键催化剂描述为“有机框架衍生物”，与参考事实中催化剂应为“蓝相铱盐”相矛盾；此前的“新型过渡金属”尚不构成明确材料错误。 |
| calculation_18x2p4m | fact | `量子` | `形成连续电子通道并降低晶界散射` | `mechanism` | 这里在“通过”之后开始给出与参考机制不一致的机制描述（量子限域效应等），构成对已知机制的显式矛盾；应从该错误机制的首个 token 起替换为正确机制。 |
| calculation_18x2p4m | constant | `k` | `0.014` | `heat_leak_coefficient` | 这里把热泄漏系数继续作为未知符号 k 来建模，而参考原子要求使用具体常数 0.014；该 token 本身就是应替换的系数占位符。 |
