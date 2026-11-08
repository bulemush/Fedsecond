# MNLI 训练与评测模板统一

MNLI 默认评测使用 `mnli_train_template_v1`，FedProxy、MA、original、proxy 和 fused 均通过公共入口 `evaluate_with_lm_eval` 使用同一协议。只要通过项目评测入口执行，配置仍写 `harness_tasks: [mnli]`，结果键也仍为 `mnli`。

训练与评测共同调用 `fedproxy.data.prompts.mnli_prompt`：

```text
Premise: {premise}
Hypothesis: {hypothesis}
Relation (entailment, neutral, or contradiction):
```

候选答案按数据集 label schema 验证为 `entailment / neutral / contradiction`。保持原有的 `validation_matched`、zero-shot、multiple-choice log-likelihood 和 accuracy；不启用 chat template，不额外加训练时的 320-token 截断。评测精度仍由原模型配置决定。其他任务继续使用原生 lm-eval 定义。

## 已训练模型无需重新训练

这次训练提示词和答案词的内容未改变，训练配置、训练预算、数据划分、压缩代理和 checkpoint 均可复用。已有 fused 导出可直接重新评测；不需要重新压缩、训练或融合。

在服务器项目根目录、原运行环境中，针对一个已有实验执行：

```bash
bash scripts/run_experiment.sh \
  runs/homogeneous_ratio30/FedProxy/llama2_homogeneous_mnli_len320_mb4_seed42/resolved_config.train.yaml \
  2 evaluate-fused
```

MA 对应使用自己的 `runs/homogeneous_ratio30/MA/.../resolved_config.train.yaml`。命令中的 GPU ID 按实际空闲显卡调整。原生 lm-eval CLI 的 `--tasks mnli` 不会自动使用项目覆盖；请通过项目入口评测。

包含 MA 与 ratio30 启动器的分支，也可以一次刷新两种方法的 MNLI：

```bash
bash scripts/run_ratio30_homogeneous_suite.sh 2,3 both mnli
```

该入口要求两个不同的可见 GPU ID。若两个实验的十轮训练和 fused 导出已经完整，检查到旧 MNLI 协议后只重新审计与评测；不会启动训练或重新融合。未完成的实验仍按启动器原有逻辑恢复训练，所以只想评测时使用上面的 `evaluate-fused` 单阶段命令。

## 结果与协议

- 旧协议评测在覆盖前复制到 `evaluation/archive/`，保留原 raw JSON 和汇总。
- 新的 raw JSON 记录 `fedproxy_task_overrides`，summary 的 `protocol.task_overrides.mnli` 记录版本、完整模板、候选答案、split 和 metric，并参与 `protocol_fingerprint`。
- 自动跳过和汇总会拒绝旧 MNLI 协议，提示重新评测，而不是复用旧分数。其他七项旧成绩可保留。
- 两种方法的八项结果均齐全后，重新生成 ratio30 汇总：

```bash
python scripts/summarize_ratio30_homogeneous_suite.py --method both
```

该成绩应标注为“MNLI 训练模板一致的评测”，与原生 `True / Neither / False` 协议的成绩分别保留。更换评测模板不构成与论文评测协议完全一致的证据。
