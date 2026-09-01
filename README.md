# FSDP Llama 训练与 BF16 指数值分析

`train.py` 仍然使用 PyTorch FSDP（`fully_shard`）训练 Llama 模型；新增的
`bf16_analysis.py` 负责在不改变训练图的情况下采集张量并统计 BF16 指数值。

## 训练模式

从零开始预训练（随机初始化）：

```bash
torchrun --nproc_per_node=8 train.py \
  --training-mode pretraining \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name <dataset> --experiment-name llama3-pretrain \
  --target-layers 0 16 31 \
  --tensor-dump-dir outputs/llama3-pretrain/tensors --capture-freq 100
```

基于本项目生成的 DCP 检查点进行后训练。`--checkpoint-dir` 指向包含
`checkpoint/` 的实验目录；如果同时提供 `--experiment-name`，新的检查点仍会
保存到新的实验目录：

```bash
torchrun --nproc_per_node=8 train.py \
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

`post-training` 也是 `posttraining` 的可用拼写。默认目标层为 `0 16 32`，可用
`--target-layers` 覆盖。采集在 backward 完成、`optimizer.zero_grad()` 之前进行，
因此每个快照同时包含：

* `weights/`：选中 Transformer 层的参数；
* `activations/`：该层输入和 forward 返回的 hidden-state（若返回 tuple，取第一个张量）；
* `gradients/`：对应参数的梯度（以及可用时的 `layer_<N>.output` 激活梯度），没有梯度的参数记为 `null`。

所有保存的张量都显式转为 BF16，并按 rank 和 step 写入
`<tensor-dump-dir>/rank-<rank>/step-XXXXXXXX.pt`。同名 `.json` 文件包含指数值窗口统计。
`--capture-freq 0`（默认值）关闭采集。

注意：标准 Meta-Llama-3-8B 配置的 `num_hidden_layers` 是 32，因此合法的
0-based 模块索引是 `0..31`；若直接使用默认的 `0 16 32`，程序会明确报告索引
越界。若“第 32 层”指最后一个模块，请按 0-based 索引传入 `--target-layers 0 16 31`；
如果使用确实包含第 32 号模块的模型，则无需修改默认值。

## 指数值窗口统计

BF16 的指数域是 IEEE-754 编码中的 8 位（数值范围 `0..255`，对应原始 BF16 bit 14
到 bit 7）。统计会在指数数值上枚举所有连续区间：宽度 3 会检查
`[0,1,2]` 到 `[253,254,255]`，宽度 7 会检查 `[0..6]` 到 `[249..255]`。每个宽度的
`top` 是包含元素比例最高的连续指数值区间（并保留并列项）；每个区间包含
`start_value`、`end_value`、`values`、`count` 和 `proportion`，`positions` 则给出所有
可能起点的完整计数。这里统计的是指数的数值区间，不是指数部分的比特模式。

直接分析一个快照或目录下的全部快照：

```bash
python analyze_bf16.py outputs/llama3-pretrain/tensors \
  --output outputs/llama3-pretrain/exponent-statistics.json
```

分析目录时，JSON 同时保留每个 rank/step 的结果，并在 `aggregated` 节点中按 step
合并各 rank 的展平元素；其中的比例对应完整逻辑张量（而不是单个 FSDP shard）。

统计包含零、非规格化数、无穷大和 NaN 的指数编码；不会静默丢弃任何元素。梯度为
`null` 的参数没有元素，因而不会出现在统计结果中。
