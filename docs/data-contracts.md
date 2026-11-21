# 数据、模型与审查接口

所有外部资产置于配置指定的 workspace，默认目录不进入 Git。

```text
workspace/
  tasks/catalog.json
  tasks/public/<task_id>/snapshot.tar
  tasks/private/<task_id>/gold.patch
  tasks/private/<task_id>/tests.patch
  tasks/private/<task_id>/source.json
  tasks/private/<task_id>/certification.json
  datasets/sft/train.jsonl
  datasets/sft/holdout.jsonl
  datasets/sft/manifest.json
  runs/<run_name>/
```

## 构造输入与来源审查

输入示例为 `examples/task-spec.example.json`。`checkout` 相对于 spec 文件目录解析；可以是绝对路径。base/fix 都会解析为 commit ID；构造器不猜测 merge parent，也不从 PR 标题自动推导正确问题。

任务需要：唯一 task_id、repo、split、defect_type、problem_statement、共享依赖 image、production_paths、test_paths、regression_tests 和 source_review。8 类缺陷的枚举位于 `tasks/schema.py`。仓库不能跨 split，任务 ID 不可重复。

来源审查应记录：issue/PR 原文来源与时间关系、审查人、必要的重构、实现提示去除、复现与预期行为，以及描述与测试的一致性。原 issue 已合格时不需要重写；与修复 PR 同期提交的报告可以合格。构造器要求明确批准记录，但不会自动判定自然语言是否泄漏答案。`patch_derived_issue` 被拒绝进入此正式构造入口，应另建候选池。

生产路径和测试路径必须覆盖所有变更文件且互不相交。构造器输出源代码快照、生产补丁和测试补丁；隐藏测试与 gold patch 仅由宿主评分器持有，不复制到 agent 容器。生产修改之外的差异会被判为不合规。对大 PR，应先人工拆出单一可验证修复任务。

认证要求非空 F2P、非空 P2P、gold 通过修复测试，并在扩大回归集上无新增失败。默认重复两次，测试 ID 和分类需一致。对日期、网络、并发等不稳定测试应先修复测试环境或排除任务，不静默放宽认证。

## 人工独立复核

`tasks freeze --human-reviews ...` 接受：

```json
{"reviews": [{"task_id": "example-repo-pr-123", "reviewer_kind": "human", "reviewer": "reviewer-id", "decision": "pass", "notes": "来源、题意和测试核对通过"}]}
```

冻结检查要求当前任务认证齐备、至少 10% 的不同任务有人类通过记录。该命令核验记录与覆盖率，不能替代实际人工审查。教师试采可在来源审查与机器认证完成后开始，不必等待最终人工冻结。

## 教师轨迹与动作审查

每条轨迹保存：

- `identity`：任务/config/策略/seed/阶段身份；
- `turns[].messages/input_ids`：本轮生成前实际可见上下文；
- `content/output_ids`：完整可见 Thought+Action；
- `execution`：命令、文件状态、原始反馈 hash、返回码、pytest 运行证据；
- `reasoning_content`：厂商返回时保留，仅用于审计，不导出为训练输入或标签；
- 最终补丁、合规结论、独立 grader 结果、终止状态和奖励分解。

`teacher audit` 输出 `action-review.jsonl`。每行需要填写 `decision=keep/reject`、`reviewer`、`reviewer_kind=human/assistant` 和 `rationale`。审查记录绑定整条轨迹文件的 SHA256；轨迹变更后旧审查失效。每个动作都需要决策，不能只审查自动标记的动作。

建议逐动作检查：动作是否基于此前信息、修改是否推进正确修复、复现是否有效、测试是否真实执行且覆盖声称范围、是否出现未验证的“已通过”声明，以及是否依赖后续才得到的信息。失败的探索动作并不自动等于坏动作，需要结合前后证据判断。

默认 `sft.whole_trajectory_only=true`：只有最终修复通过、无自动行为异常、且全部动作审查为 keep 的完整轨迹进入 SFT。设为 false 时允许保留通过轨迹中被批准的步骤，但仍保留该动作之前的原始上下文，不拼接重写历史。该选择可能改变训练分布，需作为独立实验记录。

## SFT JSONL

每行包含 task_id、repo、defect_type、trajectory_id、turn、input_ids、output_ids、source_sha256、reviewer、reviewer_kind。输入标签在训练中屏蔽，仅输出 token 参与 NLL。输入前缀需与原始 messages 经同一 tokenizer 编码后逐 token 相等。

留出集按任务抽样，不能将同一题的不同轨迹分到训练与留出。Dev/Test 任务不会进入教师采集与 SFT 构造。分层采样采用有放回抽样，因此配置中的 epoch 表示约 `样本数 × epoch` 的抽样预算，**不是每条样本恰好遍历一次**；实际抽样顺序由 seed 决定。

## 外部历史数据接入

旧项目目录中的 catalog、轨迹及报告不能直接假定与本 schema 相同。需要按上述字段转换，并在新运行环境重新认证；现有代码不会自动复制任意历史报告或将旧的成功标签当成新环境下的验证结果。仓库没有捆绑历史的 300 题资产或宣称某个教师数据版本的固定样本数。

具体数据量、覆盖和统计由 `tasks inspect` 与 `datasets/sft/manifest.json` 产生。模型和 LoRA checkpoint 通过路径配置提供，训练、推理需使用兼容的基座 revision、目标模块和 rank。
