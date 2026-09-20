# 服务器配置清单

## 软件

- Linux x86_64（建议 Ubuntu 22.04/24.04）
- Python 3.10 或 3.11
- NVIDIA driver 与 CUDA 兼容的 PyTorch；V100 使用 FP16/FP32，不启用 BF16
- 基础包：PyTorch、Transformers、PEFT、Datasets、Accelerate、safetensors、NumPy、PyYAML
- 验证包：pytest；正式评估另装 `lm-eval`
- Hugging Face 账号与 LLaMA2 权限（Mistral 通常不需要相同门控，但仍应固定 revision）
- 足够的模型缓存、run 输出和临时空间；建议至少 150 GB 可用磁盘

不在仓库里锁死 CUDA wheel。先依据服务器驱动从 PyTorch 官方选择安装命令，再执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先安装与服务器 CUDA 匹配的 torch
python -m pip install -e '.[dev,eval]'
```

## 硬件与精度

论文报告使用 4/8 张 NVIDIA V100 32 GB，但本项目 v1 是单机顺序客户端模拟，不要求多机联邦平台。7B 全模型压缩阶段仍需装载原始模型；训练阶段只装载 50% block 的代理并训练 LoRA。V100 不支持 BF16，配置 `dtype:auto` 应选择 FP16（GPU）或 FP32（CPU）。首次真实运行前应根据代理参数量、LoRA 参数量、序列长度和 micro batch 做显存探测。

建议环境变量仅用于服务器缓存和认证，例如 `HF_HOME`、`HF_TOKEN`；不要把 token 写入 YAML 或日志。

## 首次验证顺序

```bash
pytest -q
python -m fedproxy.cli smoke --config configs/smoke.yaml
python -m fedproxy.cli prepare-data --config configs/llama2_heterogeneous_50.yaml
python -m fedproxy.cli compress --config configs/llama2_heterogeneous_50.yaml
python -m fedproxy.cli train --config configs/llama2_heterogeneous_50.yaml --rounds 2
```

最后一条先作为 diagnostic；检查 loss、PCR 尺度、更新范数、截断率和层映射后再提高轮数。论文没有披露通信轮数，不能把示例轮数称为论文设置。

