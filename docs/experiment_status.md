# 实验状态

更新日期：2026-09-20。

| 项目 | 状态 | 说明 |
|---|---|---|
| 论文公式与 Algorithm 1 映射 | 已完成静态实现 | 已由 PDF 正文与附录复核 |
| Python 语法编译 | 已通过 | `python -m compileall -q src tests`，随后清理生成的 `__pycache__` |
| 11 份 YAML 配置解析与合并校验 | 已通过 | 使用现有 PyYAML，只做读取与校验 |
| CPU 单元测试 | 已编写 22 项，未运行 | 当前工作站未安装 PyTorch、PyYAML、pytest、safetensors |
| 离线 tiny Llama smoke | 未运行 | 当前工作站未安装 Transformers/PEFT/PyTorch |
| LLaMA2-7B / Mistral-7B 压缩 | 未运行 | 需要模型权限、权重、服务器内存/GPU |
| 联邦真实数据训练 | 未运行 | 需要 Hugging Face 数据缓存及服务器 GPU |
| lm-eval 对照 | 未运行 | 需要训练产物与 eval extra |

这里不预填论文数值，也不把代码实现完成等同于数值复现成功。
