# 运行、恢复与评测

所有命令使用 `repairlab --config <json> ...`。在长任务中可用 tmux 或自己的 systemd 服务保持进程；仓库不自动注册服务、不自动占用全部 GPU。

## SFT

先准备与学生 tokenizer 一致的审查后样本，再执行 `sft train --run <name>`。batch_size 是一次优化步的步骤级样本数，microbatch 为 1。默认学习率恒定，不启用推理端重复惩罚。

SFT 在配置的 epoch 比例对应的优化步保存 adapter 和优化器。启用 `evaluation.during_training` 时先评测初始策略，再逐里程碑评测相同 dev 任务。静态 holdout 数据单独导出，当前训练 runner 尚未自动计算 holdout NLL；不要把 dev 工具交互评测和静态 NLL 混为一谈。

## GRPO

`grpo.initial_adapter=null` 表示基座；给出兼容 SFT adapter 路径表示从它开始。后者的训练起点对照是该 SFT 策略，而不是另一次基座评测。

每一轮完成同策略采样、评分、概率重算、梯度更新与 checkpoint 提交。数据与课程状态随批次落盘，只有成功提交的 latest.json 决定已完成的更新。每次采样结束后才加载训练模型，训练结束后才重启 vLLM。

在同一配置下重新执行同一 run 命令会恢复已提交的优化器状态，完整轨迹和批次会检查身份后复用。若进程中断留下仅 started/partial 的轨迹，程序明确停止，要求将该未完成尝试另行归档并重新采集；不会把半条轨迹当成完整样本，也不会无提示重试计费 API。运行代码、模型或配置变化时应使用新的 run，不能覆盖原实验。

状态文件包括：`status.json`、`latest.json`、`training.json`、`sampling-progress.json`、`batches/`、`replay/`、`checkpoints/` 与 `evaluation/`。代码不会自动删除模型或历史 checkpoint。

## 单独评测

```bash
repairlab --config configs/pipeline.local.json evaluate \
  --checkpoint /path/to/initial/metadata.json \
  --output workspace/evaluations/base-dev-pass1 --split dev --k 1

repairlab --config configs/pipeline.local.json evaluate \
  --checkpoint /path/to/checkpoint/metadata.json \
  --output workspace/evaluations/checkpoint-dev-pass1 --split dev --k 1 \
  --baseline workspace/evaluations/base-dev-pass1/metrics.json
```

将 `--k 1` 改为 `--k 5` 评测首次至第五次尝试内的成功率。某题成功后不再重复该题，因此得到的是观察到的“前 k 次任一成功”比例；不会将提前停止的可变数量样本代入固定 n 的无偏组合估计公式。格式、回归和 token 统计覆盖实际执行的尝试，不是每题固定 5 条的行为分布。

对照要求相同任务集合、k、预算及采样配置；输出逐题成功→失败和失败→成功。`--split test` 才会使用 Test；Test 不用于课程或模型选择。最终发布指标前应固定 checkpoint 与预算，再执行独立 Test。

## SWE-bench Lite 接口

当前提供官方 predictions JSONL 字段的导出，不自动下载 benchmark 或执行官方 harness：

```bash
repairlab export-swebench \
  --trajectories /path/to/one-attempt-per-instance \
  --mapping /path/to/verified-instance-map.json \
  --output workspace/predictions.jsonl --model-name your-model-label
```

mapping 为 `{ "本地task_id": "官方instance_id" }`。必须是真实对应关系；不能把自建任务改名当作 SWE-bench 任务。每个 instance 只允许一条被明确选定的预测。后续需导入官方任务、适配其依赖镜像与测试协议，并使用对应版本官方评测器；本仓库不宣称这些结果已经产生。
