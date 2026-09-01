import argparse
from contextlib import contextmanager
import json
import multiprocessing
import os
import time
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint import load, save

import tqdm
import datasets
from bf16_analysis import LayerTensorCapture, analyze_snapshot
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

# fixes for reset_parameters not existing
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

def reset_rope(self: LlamaRotaryEmbedding):
    self.inv_freq, self.attention_scaling = self.rope_init_fn(
        self.config, self.inv_freq.device
    )
    self.original_inv_freq = self.inv_freq


LlamaRMSNorm.reset_parameters = lambda self: torch.nn.init.ones_(self.weight)
LlamaRotaryEmbedding.reset_parameters = reset_rope


LOGGER = logging.getLogger(__name__)

@record
def main():
    parser = _get_parser()
    args = parser.parse_args()
    # Accept both spellings in the CLI while keeping the internal branch simple.
    args.training_mode = args.training_mode.replace("-", "")

    rank = int(os.getenv("RANK", "0"))
    local_rank = rank % torch.cuda.device_count()
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(rank=rank, world_size=world_size, device_id=device)

    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    LOGGER.debug(os.environ)
    LOGGER.debug(args)
    LOGGER.debug(f"local_rank={local_rank} rank={rank} world size={world_size}")

    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    # NOTE: meta device will not allocate any memory
    model: torch.nn.Module
    with rank0_first(), torch.device("meta"):
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    LOGGER.info(
        f"Training {sum(p.numel() for p in model.parameters())} model parameters"
    )

    fsdp_config = dict(
        reshard_after_forward=True,
        offload_policy=CPUOffloadPolicy() if args.cpu_offload else None,
        mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32),
    )

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Unknown model structure")

    fsdp_layers = []
    for decoder in layers:
        fsdp_module = fully_shard(decoder, **fsdp_config)
        fsdp_layers.append(fsdp_module)

    if args.forward_prefetch_distance > 0:
        for i, fsdp_module in enumerate(fsdp_layers):
            prefetch_modules = fsdp_layers[
                i + 1 : i + 1 + args.forward_prefetch_distance
            ]
            if prefetch_modules:
                fsdp_module.set_modules_to_forward_prefetch(prefetch_modules)

    fsdp_model = fully_shard(model, **fsdp_config)

    model.to_empty(device="cpu" if args.cpu_offload else device)
    model.apply(
        lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    )
    LOGGER.info(f"Initialized model uses {get_mem_stats(device)['curr_alloc_gb']}gb")

    # NOTE: since this can download data, make sure to do the main process first
    # NOTE: This assumes that the data is on a **shared** network drive, accessible to all processes
    with rank0_first():
        train_data = _load_and_preprocess_data(args, config)
    LOGGER.debug(f"{len(train_data)} training samples")

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        # NOTE: this sampler will split dataset evenly across workers
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )
    LOGGER.debug(f"{len(dataloader)} batches per epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    is_experiment = False
    exp_dir: Path = Path(args.save_dir)
    if args.experiment_name is not None:
        is_experiment = True
        exp_dir = exp_dir / args.experiment_name

    # NOTE: full_state_dict=False means we will be saving sharded checkpoints.
    ckpt_opts = StateDictOptions(full_state_dict=False, cpu_offload=True)

    # attempt resume / load a checkpoint for post-training
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }
    resumed = False
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else exp_dir
    # Accept either an experiment directory or the DCP ``checkpoint/``
    # directory itself on the command line.
    checkpoint_id = (
        checkpoint_root
        if checkpoint_root.name == "checkpoint"
        else checkpoint_root / "checkpoint"
    )
    checkpoint_state_root = checkpoint_root.parent if checkpoint_root.name == "checkpoint" else checkpoint_root
    has_checkpoint_state = (checkpoint_state_root / "state.json").exists()
    if args.training_mode == "posttraining" and not args.checkpoint_dir and not has_checkpoint_state:
        raise ValueError(
            "posttraining requires --checkpoint-dir (a directory containing "
            "checkpoint/) or an existing experiment directory"
        )
    should_load_checkpoint = (
        bool(args.checkpoint_dir)
        or (is_experiment and has_checkpoint_state)
    )
    if should_load_checkpoint and checkpoint_id.exists():
        sharded_model_state, sharded_optimizer_state = get_state_dict(
            model, optimizer, options=ckpt_opts
        )
        load(
            dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
            checkpoint_id=checkpoint_id,
        )
        set_state_dict(
            model,
            optimizer,
            model_state_dict=sharded_model_state,
            optim_state_dict=sharded_optimizer_state,
            options=ckpt_opts,
        )
        scheduler_path = checkpoint_state_root / "lr_scheduler.pt"
        if scheduler_path.exists():
            lr_scheduler.load_state_dict(
                torch.load(scheduler_path, map_location=device, weights_only=True)
            )
        else:
            LOGGER.warning("No scheduler state at %s; using a fresh scheduler", scheduler_path)
        if has_checkpoint_state:
            with open(checkpoint_state_root / "state.json") as fp:
                state = json.load(fp)
        resumed = True
    elif should_load_checkpoint:
        raise FileNotFoundError(
            f"checkpoint directory not found: {checkpoint_id}"
        )
    if is_experiment:
        LOGGER.info(f"Resumed={resumed} | {state}")
    dist.barrier()

    if is_experiment and (
        (exp_dir.is_mount() and rank == 0)
        or (not exp_dir.is_mount() and local_rank == 0)
    ):
        LOGGER.info(f"Creating experiment root directory")
        exp_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    if is_experiment:
        (exp_dir / f"rank-{rank}").mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Worker saving to {exp_dir / f'rank-{rank}'}")

    tensor_capture = None
    if args.tensor_dump_dir:
        tensor_capture = LayerTensorCapture(model, args.target_layers)
        LOGGER.info(
            "BF16 capture enabled for layers %s (every %d step(s))",
            list(tensor_capture.target_layers),
            args.capture_freq,
        )

    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

        progress_bar = tqdm.tqdm(range(len(dataloader)), disable=rank > 0)
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        dataloader.sampler.set_epoch(state["epoch"])
        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            # NOTE: prefetches the first layer
            model.unshard()

            if tensor_capture is not None:
                tensor_capture.clear()

            with timers["data"], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            if i_step < state["epoch_step"]:
                # NOTE: for resuming
                continue

            with timers["forward"], torch.profiler.record_function("STEP::forward"):
                outputs = model(**batch)
                del batch

            with timers["backward"], torch.profiler.record_function("STEP::backward"):
                outputs.loss.backward()

            # Capture before optimizer.zero_grad() so gradients are present.
            capture_step = state["global_step"] + 1
            if (
                tensor_capture is not None
                and args.capture_freq > 0
                and capture_step % args.capture_freq == 0
            ):
                snapshot = tensor_capture.snapshot(step=capture_step)
                capture_root = Path(args.tensor_dump_dir)
                capture_path = (
                    capture_root
                    / f"rank-{rank}"
                    / f"step-{capture_step:08d}.pt"
                )
                tensor_capture.save_snapshot(snapshot, capture_path)
                stats_path = capture_path.with_suffix(".json")
                stats_path.write_text(
                    json.dumps(analyze_snapshot(snapshot), indent=2, ensure_ascii=False)
                    + "\n",
                    encoding="utf-8",
                )
                LOGGER.info("Saved BF16 tensors and exponent statistics to %s", capture_path)

            with timers["update"], torch.profiler.record_function("STEP::update"):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=not args.cpu_offload)

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += outputs.loss.item()
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
                tok_per_step = world_size * args.batch_size * args.seq_length
                ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
                info = {
                    "global_step": state["global_step"],
                    "lr": lr_scheduler.get_last_lr()[0],
                    "running_loss": state["running_loss"] / args.log_freq,
                    "epoch": state["epoch"],
                    "epoch_progress": state["epoch_step"] / len(dataloader),
                    "num_batches_remaining": len(dataloader) - i_step,
                    **get_mem_stats(device),
                    "tokens_per_s": 1000 * tok_per_step / ms_per_step,
                    "time/total": ms_per_step,
                    **{
                        f"time/{k}": timer.avg_elapsed_ms()
                        for k, timer in timers.items()
                    },
                }

                LOGGER.info(info)

                torch.cuda.reset_peak_memory_stats(device)
                state["running_loss"] = 0
                for t in timers.values():
                    t.reset()

            if is_experiment and state["global_step"] % args.ckpt_freq == 0:
                dist.barrier()
                # NOTE: we have to call this on ALL ranks
                sharded_model_state, sharded_optimizer_state = get_state_dict(
                    model, optimizer, options=ckpt_opts
                )
                save(
                    dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
                    checkpoint_id=exp_dir / "checkpoint",
                )
                if rank == 0:
                    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
                    with open(exp_dir / "state.json", "w") as fp:
                        json.dump(state, fp)
                dist.barrier()

        state["epoch_step"] = 0

    if tensor_capture is not None:
        tensor_capture.close()


def _load_and_preprocess_data(args, config):
    """
    Function created using code found in
    https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py
    """
    from itertools import chain

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data = datasets.load_dataset(args.dataset_name, args.dataset_subset)

    column_names = data["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = data.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    seq_length = args.seq_length or tokenizer.model_max_length
    if seq_length > config.max_position_embeddings:
        seq_length = min(1024, config.max_position_embeddings)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        if total_length > seq_length:
            total_length = (total_length // seq_length) * seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {seq_length}",
    )

    return lm_datasets["train"]


def get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_gb": 1e-9 * props.total_memory,
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }


@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()


class LocalTimer:
    def __init__(self, device: torch.device):
        if device.type == "cpu":
            self.synchronize = lambda: torch.cpu.synchronize(device=device)
        elif device.type == "cuda":
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
        self.measurements = []
        self.start_time = None

    def __enter__(self):
        self.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if traceback is None:
            self.synchronize()
            end_time = time.time()
            self.measurements.append(end_time - self.start_time)
        self.start_time = None

    def avg_elapsed_ms(self):
        return 1000 * (sum(self.measurements) / len(self.measurements))

    def reset(self):
        self.measurements = []
        self.start_time = None


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-name", default=None)
    parser.add_argument("-d", "--dataset-name", default=None, required=True)
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("-m", "--model-name", default=None, required=True)
    parser.add_argument("--save-dir", default="../outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=10, type=int)
    parser.add_argument("--ckpt-freq", default=500, type=int)
    parser.add_argument("-s", "--seq-length", default=1024, type=int)
    parser.add_argument("--cpu-offload", default=False, action="store_true")
    parser.add_argument("--forward-prefetch-distance", default=1, type=int)
    parser.add_argument(
        "--training-mode",
        choices=("pretraining", "posttraining", "post-training"),
        default="pretraining",
        help="initialize randomly (pretraining) or load --checkpoint-dir first",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="DCP experiment directory containing checkpoint/ (for post-training)",
    )
    parser.add_argument(
        "--tensor-dump-dir",
        default=None,
        help="directory for per-rank BF16 layer snapshots and JSON statistics",
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=[0, 16, 32],
        metavar="LAYER",
        help="Transformer layer indices to capture (default: 0 16 32)",
    )
    parser.add_argument(
        "--capture-freq",
        type=int,
        default=0,
        metavar="N",
        help="capture every N optimizer steps; 0 disables capture (default)",
    )
    return parser


if __name__ == "__main__":
    main()
