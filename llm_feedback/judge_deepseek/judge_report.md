# 多模型评审报告（LLM-as-judge）

## 评审模型

| 名称 | provider | 请求的模型 | 返回的模型 | 温度 | 调用次数 | 无效输出 | 错误 |
|---|---|---|---|---|---|---|---|
| deepseek | openai_compatible | deepseek-v4-pro | deepseek-v4-pro | 0 | 12 | 0 | 0 |

评审提示词 SHA-256：`7cc5851b4be4b77c583b6cdae48541dee413a46eb43337d95c7ee9483f38fa8c`；开始 2026-11-09T10:37:07+00:00，结束 2026-11-09T10:40:17+00:00（UTC）

## 目标：main（outputs_example_chat.json）

规则检查违规数：0；条目 16 条 × 6 项 + 运动员 6 名 × 2 项

| 评审 | 无效 | 不通过合计 | J1 | J2 | J3 | J4 | J5 | J6 | G1 | G2 |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |

多数票判为不通过的单元：2；至少一名评审判为不通过：2


## 目标：seeded（negative_control\outputs_seeded.json）

规则检查违规数：6；条目 16 条 × 6 项 + 运动员 6 名 × 2 项

| 评审 | 无效 | 不通过合计 | J1 | J2 | J3 | J4 | J5 | J6 | G1 | G2 |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek | 0 | 19 | 2 | 2 | 3 | 4 | 3 | 3 | 1 | 1 |

多数票判为不通过的单元：19；至少一名评审判为不通过：19


## 阴性对照：植入错误的检出

✓ = 在预期标准上判不通过；~ = 只在该条的其他标准上判不通过；✗ = 未发现

| 错误 | 类型 | 位置 | 内容 | 预期评审标准 | deepseek | 多数票 | 规则检查 |
|---|---|---|---|---|---|---|---|
| R1 | rule | N4 #5 | 删去最后一条问题 | G1 | ✓ | ✓ | ✓ |
| R2 | rule | N4 #1 | 度数 22° 改为 12° | J1 | ✓ | ✓ | ✓ |
| R3 | rule | N4 #2 | cue 加入清单外的右肘 | J4 | ✓ | ✓ | ✓ |
| R4 | rule | N4 #3 | looks_like 加入越界词“球速” | J6 | ✓ | ✓ | ✓ |
| R5 | rule | N3 #1 | 清空应有的 note | G2 | ✓ | ✓ | ✓ |
| S1 | semantic | N4 #4 | cue 要求加大偏差（方向反了） | J3 | ✓ | ✓ | ✗ |
| S2 | semantic | N3 #2 | looks_like 说没有偏差 | J2 | ✓ | ✓ | ✗ |
| S3 | semantic | N3 #3 | cue 把调整挪到引拍 | J4/J3 | ✓ | ✓ | ✗ |
| S4 | semantic | N3 #4 | looks_like 推断原因 | J6 | ✓ | ✓ | ✗ |
| S5 | semantic | N2 #1 | drill 改为全速实球连续击打 | J5 | ✓ | ✓ | ✗ |

阴性对照中，多数票在未植入错误的位置判为不通过的单元：0

## 需人工核对的单元（正式输出中被任一评审判为不通过）

评审模型也会误判。请逐条阅读理由：确属问题的，修改提示词后重新生成；属于误判的，在论文中如实报告。

| 目标 | 运动员 | 条 | 标准 | 不通过 / 评审数 | 多数票 | 理由 |
|---|---|---|---|---|---|---|
| main | N3 | 3 | J4 | 1/1 | 是 | deepseek: cue 和 drill 引入了躯干转动（“带动上身一起转过来”“检查骨盆和胸口是否都已转向球网”），超出骨盆转动这一关节范围 |
| main | N3 | 4 | J4 | 1/1 | 是 | deepseek: drill 涉及引拍阶段（“引拍时左臂向前伸出指向来球”），超出诊断的前挥阶段 |

## 可改写进论文的句子（英文，请核对后使用）

*Methods.* In addition to the rule-based check, 1 language models from different developers (openai_compatible deepseek-v4-pro) independently judged each output against the issue list using a fixed rubric (Supplementary Material S5): for every issue, whether the restatement was faithful (J1), the description of the movement matched the direction of the deviation (J2), the cue asked for a change opposite to the deviation (J3), the cue and drill stayed within the joint and phase (J4), the drill could be practised without a ball or at slow speed (J5) and no cause, performance or injury statement was added (J6); and for every player, whether the items corresponded one-to-one with the list (G1) and the labels and closing sentence followed the rules (G2). A criterion was judged failed when a majority of the judges failed it. Five semantic errors that the rule-based check cannot detect (a reversed cue, a description contradicting the deviation, a cue moved to another phase, an inferred cause and an unsafe drill) were seeded together with five literal errors as a negative control.

*Results.* Of the 108 judgements on the final outputs, 2 were failed by a majority of the judges and 2 by at least one judge. In the negative control, the majority vote detected 5 of 5 semantic and 5 of 5 literal errors, whereas the rule-based check detected 0 and 5, respectively.
