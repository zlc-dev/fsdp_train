torchrun --nproc_per_node=2 train.py \
  --training-mode pretraining \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name tatsu-lab/alpaca \
  --experiment-name llama3-pretrain \
  --target-layers 0 16 31 \
  --tensor-dump-dir outputs/llama3-pretrain/tensors \
  --capture-freq 100
