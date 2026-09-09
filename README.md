# FSDP Llama BF16/FP8 训练与数值分析

`train.py` 是统一训练入口，通过 `--precision bf16` 或
`--precision fp8` 选择训练精度，并包含数据、FSDP、checkpoint、训练循环和
W&B 监控逻辑。

FP8 版本通过 `torchao.float8` 将满足硬件维度约束的 `Linear` GEMM 操作量化：
forward 的 input/weight 使用硬件支持更广的 E4M3，gradient output 使用 E5M2；模型主参数、
优化器状态、FSDP 通信和其他算子仍使用 BF16。
不满足输入/输出维度 16 对齐要求的 Linear 层会自动保留 BF16。FP8 训练需要安装与
PyTorch 版本匹配的 `torchao`，并使用支持 FP8 的 GPU。

额外依赖可安装为：

```bash
pip install wandb torchao
wandb login
```

## 训练模式

从零开始预训练（随机初始化）：

```bash
torchrun --nproc_per_node=8 train.py \
  --precision bf16 \
  --training-mode pretraining \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name <dataset> --experiment-name llama3-pretrain \
  --target-layers 0 16 31 \
  --tensor-dump-dir outputs/llama3-pretrain/tensors --capture-freq 100
```

使用 FP8 训练时切换精度参数：

```bash
torchrun --nproc_per_node=8 train.py \
  --precision fp8 \
  --training-mode pretraining \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name <dataset> --experiment-name llama3-fp8 \
  --wandb-project fsdp-llama
```

基于本项目生成的 DCP 检查点进行后训练。`--checkpoint-dir` 指向包含
`checkpoint/` 的实验目录；如果同时提供 `--experiment-name`，新的检查点仍会
保存到新的实验目录：

```bash
torchrun --nproc_per_node=8 train.py \
  --precision bf16 \
  --training-mode posttraining \
  --checkpoint-dir outputs/llama3-pretrain \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name <dataset> --experiment-name llama3-posttrain \
  --target-layers 0 16 31 \
  --tensor-dump-dir outputs/llama3-posttrain/tensors --capture-freq 100
```

直接加载本地 Hugging Face `safetensors` 预训练参数进行后训练：

```bash
export HF_HUB_OFFLINE=1

torchrun --nproc_per_node=2 train.py \
  --precision bf16 \
  --training-mode posttraining \
  --model-init pretrained \
  --local-files-only \
  --model-name /data/models/Llama-3.1-8B-Instruct \
  --dataset-name <local-or-cached-dataset> \
  --experiment-name llama31-instruct-posttrain \
  --target-layers 0 16 31 \
  --batch-size 1 --seq-length 1024 \
  --tensor-dump-dir outputs/llama31-instruct-posttrain/tensors \
  --capture-freq 100
```

`--model-init pretrained` 会让 rank 0 在 CPU 中加载完整 Hugging Face 权重，再通过
FSDP2/DCP 将参数广播并切分到各 rank。模型必须是完整的 Transformers 格式目录，
至少包含 `config.json`、`model.safetensors.index.json`、所有
`model-*.safetensors` 和 tokenizer 文件。`--local-files-only` 会禁止模型配置、权重
和 tokenizer 访问 Hugging Face；数据集仍须位于本地或已经缓存。

如果同时给出有效的 `--checkpoint-dir`，DCP checkpoint 优先，用于精确恢复模型、
优化器、scheduler 和训练步数；此时不会重复加载 Hugging Face 权重。

## W&B 监控

默认只有 rank 0 创建 W&B run，并每隔 `--log-freq` 步记录跨所有 rank 平均后的
`train/loss`，以及学习率、epoch 进度、tokens/s、各训练阶段耗时和 CUDA 显存。
首次在线使用前运行 `wandb login`。常用参数如下：

```bash
--wandb-project fsdp-llama \
--wandb-entity <team> \
--wandb-run-id <stable-id> \
--wandb-mode online
```

`--wandb-run-id` 可在 checkpoint 恢复时续写同一个 run；无网络环境可使用
`--wandb-mode offline`，完全禁用则使用 `--wandb-mode disabled`。

`post-training` 也是 `posttraining` 的可用拼写。默认目标层为 `0 16 32`，可用
`--target-layers` 覆盖。采集在 backward 完成、`optimizer.zero_grad()` 之前进行，
因此每个快照同时包含：

* `weights/`：选中 Transformer 层的参数；
* `activations/`：该层输入和 forward 返回的 hidden-state（若返回 tuple，取第一个张量）；
* `gradients/`：对应参数的梯度（以及可用时的 `layer_<N>.output` 激活梯度），没有梯度的参数记为 `null`。

保存的张量保持采集时的原始 dtype，并按 rank 和 step 写入
`<tensor-dump-dir>/rank-<rank>/step-XXXXXXXX.pt`。训练阶段不做任何 FP16/BF16/FP8
数值统计；FP8 E5M2 转换和统计请在单独的离线分析项目中完成。
`--capture-freq 0`（默认值）关闭采集。

注意：标准 Meta-Llama-3-8B 配置的 `num_hidden_layers` 是 32，因此合法的
0-based 模块索引是 `0..31`；若直接使用默认的 `0 16 32`，程序会明确报告索引
越界。若“第 32 层”指最后一个模块，请按 0-based 索引传入 `--target-layers 0 16 31`；
如果使用确实包含第 32 号模块的模型，则无需修改默认值。

本项目不再提供离线统计入口；请在单独的离线分析项目中读取这些 `.pt` 快照。
