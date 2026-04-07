"""
MiniMind 预训练（Accelerate + 可选 FSDP）

核心流程对齐 `trainer/train_pretrain.py`：余弦学习率、CE+MoE aux loss、梯度累积与裁剪、
权重默认写入项目根下 `out_v2/`；resume 与 `lm_checkpoint` 兼容文件在 `checkpoints_v2/`。勿使用旧版 `out/`、`checkpoints/`，避免覆盖原 trainer 产物。

单卡:
  python trainer_v2/train_pretrain_v2.py

多卡 DDP / FSDP（推荐）:
  accelerate launch trainer_v2/train_pretrain_v2.py
  accelerate launch trainer_v2/train_pretrain_v2.py --use_fsdp 1
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
import shutil
import subprocess
import sys
import time
import warnings
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_OUT_V2_DIR = os.path.join(_ROOT_DIR, "out_v2")
_CHECKPOINTS_V2_DIR = os.path.join(_ROOT_DIR, "checkpoints_v2")
_LOGS_V2_DIR = os.path.join(_ROOT_DIR, "logs_v2")
_CONFIG_PRETRAIN_DIR = os.path.join(_ROOT_DIR, "config", "pretrain")
_DEFAULT_CONFIG_PATH = os.path.join(_CONFIG_PRETRAIN_DIR, "config.yaml")
_DEFAULT_TOKENIZER_PATH = os.path.join(_ROOT_DIR, "model")
_DEFAULT_DATA_PATH = os.path.join(_ROOT_DIR, "dataset", "pretrain_hq.jsonl")
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler

from accelerate import Accelerator
from accelerate.utils import DistributedType, FullyShardedDataParallelPlugin

try:
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
except ImportError:
    StateDictType = None
    FullStateDictConfig = None

from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset
from model.model_minimind_v2 import MiniMindConfigV2, MiniMindForCausalLMV2
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    get_model_params,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")


class _TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _get_launch_dir(args: argparse.Namespace) -> str:
    env_dir = os.environ.get("MM_LAUNCH_DIR")
    if env_dir:
        return env_dir
    stage = getattr(args, "stage", "pretrain_v2")
    run_tag = getattr(args, "run_tag", "") or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return os.path.join(getattr(args, "logs_dir", _LOGS_V2_DIR), stage, run_tag)


def _setup_process_logging(args: argparse.Namespace, process_index: int, is_main_process: bool) -> None:
    if int(getattr(args, "log_to_file", 1)) != 1:
        return
    launch_dir = _get_launch_dir(args)
    os.makedirs(launch_dir, exist_ok=True)
    single_file = int(getattr(args, "log_single_file", 1)) == 1

    # 多卡默认仅保留 rank0 一个日志文件，避免目录里大量分卡日志。
    if single_file and not is_main_process:
        devnull = open(os.devnull, "w", encoding="utf-8")
        if int(getattr(args, "log_to_terminal", 0)) == 1:
            # 保留终端输出（通常也会很少），文件端丢弃
            sys.stderr = _TeeWriter(sys.__stderr__, devnull)
            sys.stdout = _TeeWriter(sys.__stdout__, devnull)
        else:
            sys.stdout = devnull
            sys.stderr = devnull
        return

    if is_main_process:
        with open(os.path.join(launch_dir, "run.meta"), "a", encoding="utf-8") as mf:
            mf.write(f"started_at={datetime.now().isoformat()}\n")
            mf.write(f"cwd={os.getcwd()}\n")
            mf.write(f"argv={' '.join(sys.argv)}\n")
    log_file = open(os.path.join(launch_dir, f"rank{process_index}.log"), "a", encoding="utf-8", buffering=1)
    if int(getattr(args, "log_to_terminal", 0)) == 1:
        sys.stdout = _TeeWriter(sys.__stdout__, log_file)
        sys.stderr = _TeeWriter(sys.__stderr__, log_file)
    else:
        sys.stdout = log_file
        sys.stderr = log_file


def _ensure_nccl_env_for_rtx4000(args: argparse.Namespace) -> None:
    """
    RTX 40 系列在多卡/FSDP 场景下，accelerate 可能要求关闭 P2P/IB。
    `accelerate launch` 会自动设置；直接 `python` 启动时这里做兜底。
    """
    if not torch.cuda.is_available():
        return
    multi_gpu = int(getattr(args, "num_processes", 1)) > 1
    if not (multi_gpu or int(getattr(args, "use_fsdp", 0)) == 1):
        return
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")


def _load_training_config_defaults(config_path: str) -> dict:
    """
    从 YAML 配置文件读取默认值，支持工业化分组：
      training / distributed / data / model / checkpoint / logging / runtime
    也兼容 legacy 的 training.*。
    """
    if not config_path:
        return {}
    cfg_path = os.path.expanduser(config_path)
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.normpath(os.path.join(_ROOT_DIR, cfg_path))
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    sections = (
        "training",
        "distributed",
        "data",
        "model",
        "checkpoint",
        "logging",
        "runtime",
        "monitoring",
        "optimization",
    )
    defaults: dict = {}
    for sec in sections:
        block = data.get(sec, {})
        if isinstance(block, dict):
            defaults.update(block)
    # 允许少量顶层直配键（会覆盖分组值）
    for k, v in data.items():
        if not isinstance(v, dict):
            defaults[k] = v
    return defaults


def resolve_local_pretrained_path(path: str) -> str:
    """
    本地 tokenizer / 模型目录转为绝对路径。
    避免 '../model' 等相对路径被 Hugging Face Hub 误判为 repo id（HFValidationError）。
    """
    path = os.path.expanduser(path.strip())
    if os.path.isabs(path):
        return os.path.normpath(path)
    # 与旧 trainer 一致：在仓库根目录运行时 ../model 应对应本仓库的 model/，而非上级目录
    norm = path.replace("\\", "/")
    if norm in ("../model", "model", "./model"):
        legacy = os.path.join(_ROOT_DIR, "model")
        if os.path.isdir(legacy):
            return os.path.normpath(legacy)
    for base in (_ROOT_DIR, os.getcwd()):
        cand = os.path.normpath(os.path.join(base, path))
        if os.path.isdir(cand) or os.path.isfile(cand):
            return cand
    return os.path.normpath(os.path.join(os.getcwd(), path))


def resolve_data_file(path: str) -> str:
    path = os.path.expanduser(path.strip())
    if os.path.isabs(path):
        return os.path.normpath(path)
    for base in (_ROOT_DIR, os.getcwd()):
        cand = os.path.normpath(os.path.join(base, path))
        if os.path.isfile(cand):
            return cand
    return os.path.normpath(os.path.join(os.getcwd(), path))


def _save_checkpoint_accelerate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lm_config: MiniMindConfigV2,
    weight: str,
    epoch: int,
    step: int,
    save_dir: str,
    checkpoints_dir: str,
    wandb_module,
    scaler: torch.cuda.amp.GradScaler | None = None,
):
    """
    与 `lm_checkpoint` 相同产物：{save_dir}/{weight}_{hidden}.pth 与 checkpoints_v2 下 resume.pth（save_dir 默认 out_v2）。
    FSDP 下用 `accelerator.get_state_dict` 聚合整模。
    """
    import torch.distributed as dist

    raw_sd = accelerator.get_state_dict(model)
    if not accelerator.is_main_process:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    moe_suffix = "_moe" if lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    resume_path = f"{checkpoints_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}_resume.pth"

    state_dict = {k: v.half().cpu() for k, v in raw_sd.items()}
    ckp_tmp = ckp_path + ".tmp"
    torch.save(state_dict, ckp_tmp)
    os.replace(ckp_tmp, ckp_path)

    wandb_id = None
    if wandb_module is not None:
        if hasattr(wandb_module, "get_run"):
            run = wandb_module.get_run()
            wandb_id = getattr(run, "id", None) if run else None
        else:
            wandb_id = getattr(wandb_module, "id", None)

    resume_data = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "wandb_id": wandb_id,
    }
    if scaler is not None:
        resume_data["scaler"] = scaler.state_dict()

    resume_tmp = resume_path + ".tmp"
    torch.save(resume_data, resume_tmp)
    os.replace(resume_tmp, resume_path)
    del state_dict, resume_data, raw_sd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_epoch(
    accelerator: Accelerator,
    epoch: int,
    loader: DataLoader,
    iters: int,
    start_step: int,
    wandb_module,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lm_config: MiniMindConfigV2,
    scaler: torch.cuda.amp.GradScaler | None,
    trial=None,
    optuna_report_step: list[int] | None = None,
    optimizer_step_counter: list[int] | None = None,
    token_counter: list[int] | None = None,
) -> tuple[float | None, bool]:
    """trial 非空时向 Optuna 报告 loss 并支持剪枝；optuna_report_step 为跨 epoch 递增的 list[int]（如 [0]）。"""
    start_time = time.time()
    last_loss: float | None = None
    max_steps = getattr(args, "max_train_steps", 0) or 0
    max_optimizer_steps = getattr(args, "max_optimizer_steps", 0) or 0
    target_tokens = getattr(args, "target_tokens", 0) or 0
    # 仅用于日志 ETA 展示，避免 max_train_steps 生效时分母仍显示完整 epoch 步数
    effective_total_steps = iters
    if max_steps > 0:
        remaining = max_steps - epoch * iters
        if remaining > 0:
            effective_total_steps = min(iters, remaining)
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        global_step = epoch * iters + step
        input_ids = input_ids.to(accelerator.device, non_blocking=True)
        labels = labels.to(accelerator.device, non_blocking=True)
        # 与 CausalLM loss 对齐：按 shift 后有效标签计 token（排除 -100）
        step_tokens = (labels[:, 1:] != -100).sum().to(accelerator.device)
        step_tokens = accelerator.reduce(step_tokens, reduction="sum").item()
        if token_counter is not None:
            token_counter[0] += int(step_tokens)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with accelerator.accumulate(model):
            outputs = model(input_ids, labels=labels)
            aux = outputs.aux_loss if outputs.aux_loss is not None else input_ids.new_zeros(())
            loss = (outputs.loss + aux) / args.accumulation_steps
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if optimizer_step_counter is not None:
                    optimizer_step_counter[0] += 1

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.detach().float().item() * args.accumulation_steps
            aux_val = aux.detach().float().item() if torch.is_tensor(aux) else float(aux)
            logits_loss = current_loss - aux_val
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / max(step - start_step, 1) * max(effective_total_steps - step, 0) // 60
            step_view_total = effective_total_steps if max_steps > 0 else iters
            opt_steps_now = optimizer_step_counter[0] if optimizer_step_counter is not None else 0
            opt_steps_txt = (
                f", optimizer_step: {opt_steps_now}/{max_optimizer_steps}"
                if max_optimizer_steps > 0
                else ""
            )
            tokens_now = token_counter[0] if token_counter is not None else 0
            tokens_txt = f", tokens: {tokens_now}/{target_tokens}" if target_tokens > 0 else ""
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{step_view_total}), "
                f"loss: {current_loss:.4f}, logits_loss: {logits_loss:.4f}, aux_loss: {aux_val:.4f}, "
                f"lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min{opt_steps_txt}{tokens_txt}"
            )
            last_loss = float(current_loss)
            if trial is not None and accelerator.is_main_process:
                import optuna

                if optuna_report_step is not None:
                    optuna_report_step[0] += 1
                    rstep = optuna_report_step[0]
                else:
                    rstep = global_step
                trial.report(last_loss, rstep)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if wandb_module is not None and accelerator.is_main_process:
                wandb_module.log(
                    {
                        "loss": current_loss,
                        "logits_loss": logits_loss,
                        "aux_loss": aux_val,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        if (not getattr(args, "no_save_checkpoint", False)) and (
            step % args.save_interval == 0 or step == iters
        ):
            # FSDP 下 get_state_dict 需所有 rank 参与；写文件仅在主进程
            model.eval()
            _save_checkpoint_accelerate(
                accelerator,
                model,
                optimizer,
                lm_config,
                args.save_weight,
                epoch,
                step,
                args.save_dir,
                _CHECKPOINTS_V2_DIR,
                wandb_module,
                scaler=scaler,
            )
            model.train()

        del input_ids, labels, outputs, loss

        if max_steps > 0 and global_step >= max_steps:
            return last_loss, True
        if (
            max_optimizer_steps > 0
            and optimizer_step_counter is not None
            and optimizer_step_counter[0] >= max_optimizer_steps
        ):
            return last_loss, True
        if target_tokens > 0 and token_counter is not None and token_counter[0] >= target_tokens:
            return last_loss, True

    return last_loss, False


def build_fsdp_plugin(args: argparse.Namespace) -> FullyShardedDataParallelPlugin | None:
    if not args.use_fsdp:
        return None
    kwargs: dict = {
        "use_orig_params": True,
        "sync_module_states": True,
        "transformer_cls_names_to_wrap": ["MiniMindBlock"],
    }
    if StateDictType is not None and FullStateDictConfig is not None:
        kwargs["state_dict_type"] = StateDictType.FULL_STATE_DICT
        kwargs["state_dict_config"] = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
    if args.fsdp_sharding == "full_shard":
        kwargs["sharding_strategy"] = "FULL_SHARD"
    elif args.fsdp_sharding == "shard_grad_op":
        kwargs["sharding_strategy"] = "SHARD_GRAD_OP"
    elif args.fsdp_sharding == "no_shard":
        kwargs["sharding_strategy"] = "NO_SHARD"
    return FullyShardedDataParallelPlugin(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniMind Pretraining (Accelerate / FSDP)")
    p.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG_PATH,
        help="YAML 配置文件路径（读取 training/distributed/data/model/checkpoint/logging/runtime 作为默认值）",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default=_OUT_V2_DIR,
        help="模型权重保存目录（默认项目根 out_v2，避免覆盖旧 trainer 的 out/）",
    )
    p.add_argument("--save_weight", default="pretrain", type=str, help="保存权重前缀")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--hidden_size", default=768, type=int)
    p.add_argument("--num_hidden_layers", default=8, type=int)
    p.add_argument("--max_seq_len", default=340, type=int)
    p.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    p.add_argument(
        "--data_path",
        type=str,
        default=_DEFAULT_DATA_PATH,
        help="预训练 jsonl（默认项目根 dataset/ 下）",
    )
    p.add_argument("--from_weight", default="none", type=str)
    p.add_argument(
        "--from_weight_dir",
        type=str,
        default=None,
        help="初始化权重目录，默认与 --save_dir 相同；可从旧版 out/ 指定路径，保存仍用 out_v2",
    )
    p.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    p.add_argument(
        "--tokenizer_path",
        type=str,
        default=_DEFAULT_TOKENIZER_PATH,
        help="本地 tokenizer 目录（默认项目根 model/，须为绝对路径或相对项目根/当前目录存在的路径）",
    )
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    p.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    p.add_argument("--use_fsdp", default=0, type=int, choices=[0, 1], help="使用 Accelerate FSDP（多卡时建议开启）")
    p.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="自动 launch 时使用的进程数（默认 1 卡；显式指定 4 才走 4 卡）",
    )
    p.add_argument(
        "--auto_accelerate_launch",
        type=int,
        default=1,
        choices=[0, 1],
        help="直接 python 启动时是否自动转为 accelerate launch（仅 num_processes>1 时生效）",
    )
    p.add_argument(
        "--fsdp_sharding",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "no_shard"],
        help="FSDP sharding 策略",
    )
    p.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help=">0 时在达到该全局 step 后结束训练（用于 Optuna 等快速试验）",
    )
    p.add_argument(
        "--max_optimizer_steps",
        type=int,
        default=0,
        help=">0 时在达到该 optimizer step 后结束训练（不受 batch_size/max_train_steps 间接影响）",
    )
    p.add_argument(
        "--target_tokens",
        type=int,
        default=0,
        help=">0 时在达到该全局 token 数后结束训练（跨所有 GPU 求和，按有效 label token 计）",
    )
    p.add_argument("--stage", type=str, default="pretrain_v2", help="训练阶段名（用于日志目录分层）")
    p.add_argument("--run_tag", type=str, default="", help="本次启动标签（默认自动时间戳）")
    p.add_argument("--logs_dir", type=str, default=_LOGS_V2_DIR, help="日志根目录")
    p.add_argument("--log_to_file", type=int, default=1, choices=[0, 1], help="是否写入日志文件")
    p.add_argument("--log_to_terminal", type=int, default=0, choices=[0, 1], help="是否同时打印到终端")
    p.add_argument(
        "--log_single_file",
        type=int,
        default=1,
        choices=[0, 1],
        help="多卡时是否仅保留主进程一个日志文件（1=是，0=每个 rank 一个）",
    )
    return p


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    pre_args, _ = parser.parse_known_args()
    cfg_defaults = _load_training_config_defaults(getattr(pre_args, "config", ""))
    if cfg_defaults:
        parser.set_defaults(**cfg_defaults)
    return parser.parse_args()


def _strip_launcher_args(argv: list[str]) -> list[str]:
    """移除本脚本自动 launch 控制参数，避免转发后冲突。"""
    filtered: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--auto_accelerate_launch":
            i += 2
            continue
        if a.startswith("--auto_accelerate_launch="):
            i += 1
            continue
        if a == "--num_processes":
            i += 2
            continue
        if a.startswith("--num_processes="):
            i += 1
            continue
        filtered.append(a)
        i += 1
    return filtered


def maybe_relaunch_distributed(args: argparse.Namespace) -> None:
    """
    直接 `python train_pretrain_v2.py ...` 时，若 num_processes>1 自动切到对应卡数的 accelerate launch。
    已在分布式上下文（RANK/LOCAL_RANK）中时不再重启，避免递归。
    """
    if int(getattr(args, "auto_accelerate_launch", 1)) != 1:
        return
    if "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        return
    if not torch.cuda.is_available():
        return
    need = int(getattr(args, "num_processes", 1))
    if need <= 1:
        return
    have = torch.cuda.device_count()
    if have < need:
        raise RuntimeError(f"可见 GPU 数不足：需要 {need} 张，当前仅 {have} 张。")

    accelerate_bin = shutil.which("accelerate")
    if accelerate_bin is None:
        raise RuntimeError("未找到 accelerate 命令，请先安装并确保其在 PATH 中。")

    forwarded = _strip_launcher_args(sys.argv[1:])
    launch_dir = _get_launch_dir(args) if int(getattr(args, "log_to_file", 1)) == 1 else ""
    cmd = [
        accelerate_bin,
        "launch",
        "--num_processes",
        str(need),
        os.path.abspath(__file__),
        "--auto_accelerate_launch",
        "0",
    ] + forwarded
    env = os.environ.copy()
    if launch_dir:
        env["MM_LAUNCH_DIR"] = launch_dir
    Logger(f"检测到单进程启动，自动切换为 {need} 卡: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)
    raise SystemExit(0)


def run_pretrain(args: argparse.Namespace, trial=None) -> float:
    """执行完整训练流程；返回最后一次日志记录的 loss（无则 inf）。"""
    args.tokenizer_path = resolve_local_pretrained_path(args.tokenizer_path)
    args.data_path = resolve_data_file(args.data_path)
    _ensure_nccl_env_for_rtx4000(args)
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(
            f"预训练数据文件不存在: {args.data_path}\n"
            f"请确认路径，或使用 --data_path 指向 dataset 下已有 jsonl（如 pretrain_hq.jsonl）。"
        )

    mixed_precision = None
    if args.dtype == "bfloat16":
        mixed_precision = "bf16"
    elif args.dtype == "float16":
        mixed_precision = "fp16"

    fsdp_plugin = build_fsdp_plugin(args)
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.accumulation_steps,
        fsdp_plugin=fsdp_plugin,
    )
    _setup_process_logging(args, accelerator.process_index, accelerator.is_main_process)

    setup_seed(42 + accelerator.process_index)

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfigV2(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=_CHECKPOINTS_V2_DIR)
        if args.from_resume == 1
        else None
    )

    wandb = None
    if args.use_wandb and accelerator.is_main_process:
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        name = (
            f"MiniMind-Pretrain-v2-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=name, id=wandb_id, resume=resume)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLMV2(lm_config)

    if args.from_weight != "none":
        moe_suffix = "_moe" if lm_config.use_moe else ""
        wdir = args.from_weight_dir if args.from_weight_dir is not None else args.save_dir
        wdir = resolve_local_pretrained_path(wdir)
        weight_path = f"{wdir}/{args.from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        weights = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(weights, strict=False)

    if ckp_data is not None:
        model.load_state_dict(ckp_data["model"], strict=False)

    get_model_params(model, lm_config)
    Logger(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")

    model = model.to(accelerator.device)

    if args.use_compile == 1 and not args.use_fsdp:
        model = torch.compile(model)
        Logger("torch.compile enabled")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=getattr(args, "weight_decay", 0.0)
    )

    start_epoch, start_step = 0, 0
    scaler = None
    if ckp_data is not None:
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        saved_ws = int(ckp_data.get("world_size", 1))
        current_ws = int(accelerator.num_processes)
        world_size_changed = saved_ws != current_ws
        # FSDP 下优化器状态与单卡 checkpoint 形状可能不一致，失败时仅恢复权重
        if world_size_changed:
            Logger(
                f"resume: 检测到 world_size 变化({saved_ws}->{current_ws})，"
                "为避免优化器状态形状不兼容，跳过 optimizer 恢复，仅恢复模型与步数。"
            )
        else:
            try:
                optimizer.load_state_dict(ckp_data["optimizer"])
            except Exception as e:
                if accelerator.distributed_type == DistributedType.FSDP:
                    Logger(f"resume: 跳过 optimizer 恢复（FSDP 与旧 checkpoint 可能不兼容）: {e}")
                else:
                    Logger(f"resume: optimizer 恢复失败，自动跳过并继续训练: {e}")

    model, optimizer = accelerator.prepare(model, optimizer)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    if getattr(args, "max_train_steps", 0) and int(args.max_train_steps) > 0:
        Logger(f"max_train_steps enabled: {args.max_train_steps} (按 data loader step 计数)")
    if getattr(args, "max_optimizer_steps", 0) and int(args.max_optimizer_steps) > 0:
        Logger(f"max_optimizer_steps enabled: {args.max_optimizer_steps} (按 optimizer step 计数)")
    if getattr(args, "target_tokens", 0) and int(args.target_tokens) > 0:
        Logger(f"target_tokens enabled: {args.target_tokens} (跨所有 GPU 的有效 token 总数)")

    final_loss = float("inf")
    optuna_report_step = [0] if trial is not None else None
    optimizer_step_counter = [0]
    token_counter = [0]
    for epoch in range(start_epoch, args.epochs):
        accelerator.wait_for_everyone()
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()

        if accelerator.num_processes > 1:
            train_sampler = DistributedSampler(train_ds, shuffle=True)
            train_sampler.set_epoch(epoch)
            skip = start_step if (epoch == start_epoch and start_step > 0) else 0
            batch_sampler = SkipBatchSampler(train_sampler, args.batch_size, skip)
        else:
            train_sampler = None
            skip = start_step if (epoch == start_epoch and start_step > 0) else 0
            batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)

        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        if skip > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前 {start_step} 个 step，从 step {start_step + 1} 开始"
            )
            iters = len(loader) + skip
            eloss, stop_all = train_epoch(
                accelerator,
                epoch,
                loader,
                iters,
                start_step,
                wandb,
                args,
                model,
                optimizer,
                lm_config,
                scaler,
                trial=trial,
                optuna_report_step=optuna_report_step,
                optimizer_step_counter=optimizer_step_counter,
                token_counter=token_counter,
            )
        else:
            iters = len(loader)
            eloss, stop_all = train_epoch(
                accelerator,
                epoch,
                loader,
                iters,
                0,
                wandb,
                args,
                model,
                optimizer,
                lm_config,
                scaler,
                trial=trial,
                optuna_report_step=optuna_report_step,
                optimizer_step_counter=optimizer_step_counter,
                token_counter=token_counter,
            )

        if eloss is not None:
            final_loss = eloss
        start_step = 0
        if stop_all:
            break

    accelerator.wait_for_everyone()
    # 显式结束训练，避免 PyTorch 2.4+ 的 ProcessGroupNCCL 未销毁告警
    if hasattr(accelerator, "end_training"):
        accelerator.end_training()
    elif dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if wandb is not None and accelerator.is_main_process and hasattr(wandb, "finish"):
        wandb.finish()

    return final_loss


def main():
    args = parse_args()
    maybe_relaunch_distributed(args)
    run_pretrain(args)


if __name__ == "__main__":
    main()
