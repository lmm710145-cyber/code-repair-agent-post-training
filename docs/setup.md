# 安装与资源

建议 Python 3.11 + Linux + NVIDIA GPU + Docker。任务容器运行时禁网；依赖必须预先构建进镜像。Docker 镜像和代码快照是两个独立资产。

## 仅使用核心工具

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
repairlab doctor
pytest -q
```

核心安装不下载模型、不安装 Torch/vLLM，也不调用教师。`doctor` 只检查环境，不启动 GPU 模型。需要采集教师或准备真实 SFT 数据时安装 `pip install -e '.[client]'`，用于加载与学生一致的 tokenizer。

## 单卡训练与推理分环境

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
# 根据本机驱动安装匹配的 PyTorch/CUDA wheel，再安装以下依赖。
pip install -r requirements/train.txt
pip install -e .
deactivate

python3.11 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -r requirements/serve.txt
pip install -e .
deactivate

source .venv-train/bin/activate
export REPAIRLAB_MODEL=/absolute/path/to/Qwen3.5-9B
export REPAIRLAB_VLLM_PYTHON="$PWD/.venv-vllm/bin/python"
```

版本参考来自原始实验环境：训练 Torch 2.14.1+cu130、Unsloth/Unsloth Zoo 2026.11.3、Transformers 5.17.0、PEFT 0.21.2；推理 vLLM 0.33.0、Torch 2.14.0；mini-SWE-agent 2.4.6。requirements 固定主要 Python 组件，但不构成所有操作系统、CUDA 与可选内核的完整锁定。新机器需要检查依赖解析和显存容量。

默认 BF16、不量化、语言模块 LoRA r16，参考目标为约 32 GiB 单卡。vLLM 和训练 worker 交替运行，不能同时加载两份模型。`vllm.memory_fraction`、并发数、预算都在配置中；并发减少不会改变 G。

模型默认 `local_files_only=true`，路径通过环境变量给出。远程模型下载和访问授权由使用者另行完成。可设置 `model.revision` 固定 Hugging Face revision；对本地快照建议保存外部权重校验清单。

## 依赖镜像

`docker/Dockerfile.tasks` 仅为合成任务和通用 Python 测试提供基础工具。真实仓库通常需要各自的依赖版本：

1. 从模板派生共享镜像，在构建阶段安装锁定依赖。
2. 将实际镜像 tag 或 digest 写入 task spec 的 `image`。
3. `setup_commands` 只做离线、可重复的任务准备，例如 editable install；准备过程不能修改受跟踪代码。
4. 执行 `tasks certify`。证书保存镜像 ID 与任务摘要，镜像或任务变化后必须重新认证。

代码使用当前用户的 Docker 权限，不自动 sudo、不更改用户组。任务容器不挂载宿主目录或密钥。Docker 隔离、补丁合规检查和 pytest 观测器用于实验控制，不应被理解为对任意恶意代码的完备安全证明。
