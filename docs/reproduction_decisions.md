# FedProxy 复现决策

以下项目分为“论文明确设定”和“复现实现约定”。后者不是作者源码事实，运行时应与 resolved config 一起留档。

| ID | 第一版选择 | 类型 |
|---|---|---|
| D01 | 任务向量、PCR、H-TIES 均定义在 LoRA A/B 参数空间 | 推断实现 |
| D02 | `remove_ratio` 是删除 block 的请求比例，同时记录真实参数删除比例 | 实现约定 |
| D03 | `keep=max(1, L-floor(L*remove_ratio))` | 实现约定 |
| D04 | 不强制保留首尾 block；embedding、final norm、lm_head 始终保留 | 实现约定 |
| D05 | LoRA 覆盖 q/k/v/o 与 gate/up/down projection | 实现约定 |
| D06 | PCR 按式 (7)、(9)：冲突越高，正则越强；正文相反描述视为笔误 | 公式优先 |
| D07 | 式 (6) 保留绝对 cosine，负相关也可获得高权重 | 论文明确公式 |
| D08 | 全零坐标按式 (7) 得到冲突 1 | 论文明确公式 |
| D09 | `max(h)-min(h)<=1e-12` 时 `h_norm=0` | 数值约定 |
| D10 | 稀疏化默认对完整客户端 LoRA 向量做 global top-k | 实现约定 |
| D11 | `m=floor(rP)`，同幅度按规范参数名和展平索引稳定打破并列 | 实现约定 |
| D12 | 通信轮数论文未披露，真实训练必须显式指定 | 未披露 |
| D13 | local epochs=10 解释为每轮 10 epoch | 论文值 + 实现解释 |
| D14 | 每客户端每轮重置 AdamW | 实现约定 |
| D15 | 使用 `unified_v1` 非 chat 模板和 response-only CE | 实现约定 |
| D16 | 训练集超过 5000 条时先固定抽样再划分 | 论文明确 |
| D17 | Alpaca 校准默认 512 条、seed=42 | 实现约定 |
| D18 | constant LR、weight decay=0 | 实现约定 |
| D19 | PCR 对所有 LoRA 参数求和，不改成均值 | 论文明确公式 |
| D20 | 分析与聚合用 FP32，epsilon=1e-12 | 数值约定 |
| D21 | batch size 16 解释为客户端有效 batch | 论文值 + 实现解释 |
| D22 | 首版所有客户端每轮全参与 | Algorithm 1 |
| D23 | lm-eval 默认 zero-shot；明确记录 split、metric 与模板版本 | 实现约定 |

论文摘要使用了“distilling”措辞，但 §4.1 只给出了 BI 结构化层剪枝，没有 KL、教师 logits 或额外蒸馏训练，因此首版不实现也不宣称存在蒸馏优化。

