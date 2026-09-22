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

论文报告使用 4/8 张 NVIDIA V100 32 GB；本项目采用单机联邦模拟，支持单卡顺序客户端训练和多卡客户端并行，不要求多机联邦平台。7B 全模型压缩阶段仍需装载原始模型；训练阶段每张工作 GPU 装载一份 50% block 的代理并训练对应客户端的 LoRA。V100 不支持 BF16，配置 `dtype:auto` 应选择 FP16（GPU）或 FP32（CPU）。首次真实运行前应根据代理参数量、LoRA 参数量、序列长度和 micro batch 做显存探测。

建议环境变量仅用于服务器缓存和认证，例如 `HF_HOME`、`HF_TOKEN`；不要把 token 写入 YAML 或日志。

项目数据目录由 `data.local_dir` 控制，默认是仓库下的 `data/`。每个数据集会先从该目录的 `datasets/` 子目录读取；只有本地副本不存在时才访问 Hugging Face，并在下载后保存可离线重载的副本。服务器推荐覆盖到容量充足的数据盘，例如：

```bash
--set data.local_dir=/home/async/data-disk/wgh/FedProxy-data
```

也可以预先将本地数据放成 `data/<任务>/<split>.jsonl`、`.json`、`.csv`、`.parquet`，或使用 Hugging Face `Dataset.save_to_disk()` 保存到 `data/<任务>/<split>/`。首次读取后会统一转换到 `data/datasets/<任务>/<revision>/<split>/`。

## 单卡与多卡入口

无需在每次登录后重新定义 Bash 数组。实验参数全部保存在 `configs/experiments/`，Shell 入口的第二个参数只负责选择物理 GPU：

```bash
./scripts/run_experiment.sh configs/experiments/llama2_medium.yaml 3
./scripts/run_experiment.sh configs/experiments/llama2_large_multigpu.yaml 0,1
./scripts/run_experiment.sh configs/experiments/llama2_full_multigpu.yaml 0,1,2,3
```

Xshell 断开后仍需继续运行时，改用后台入口：

```bash
./scripts/launch_experiment.sh configs/experiments/llama2_large_multigpu.yaml 2,3 train
```

它会把输出保存到 `runs/<实验名>/logs/`，并打印可直接执行的 `tail -f` 命令；PID 同时写入同目录的 `<stage>.pid`。

`CUDA_VISIBLE_DEVICES=2,3` 后，工作进程内部看到的设备编号会重新映射为 `cuda:0` 和 `cuda:1`，日志中的 `visible_device_index` 指的是这个可见编号。多卡实现并行训练不同客户端，而不是让不同客户端串行共享一个 DDP 模型；这保留了客户端模型、优化器和数据的隔离。

多进程边界上的 LoRA adapter、round base 和 conflict state 都会先打包为单个连续张量，再通过共享内存传输；不会为数百个 LoRA 参数分别打开文件描述符，因此无需依赖提高 `ulimit -n`。

`dry-run` 只解析数据和预算，不装载训练模型。默认 `evaluate` 阶段依次生成 original、proxy、fused 三套结果；若只想补跑一种模型，可使用 `evaluate-original`、`evaluate-proxy` 或 `evaluate-fused`。

## 首次验证顺序

```bash
pytest -q
python -m fedproxy.cli smoke --config configs/smoke.yaml
python -m fedproxy.cli prepare-data --config configs/llama2_heterogeneous_50.yaml
python -m fedproxy.cli compress --config configs/llama2_heterogeneous_50.yaml
python -m fedproxy.cli train --config configs/llama2_heterogeneous_50.yaml --rounds 2
```

最后一条先作为 diagnostic；检查 loss、PCR 尺度、更新范数、截断率和层映射后再提高轮数。论文没有披露通信轮数，不能把示例轮数称为论文设置。
