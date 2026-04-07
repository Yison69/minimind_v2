"""
MiniMind 全参数 SFT（TRL SFTTrainer + Transformers）

核心目标：
- 保留旧版 `trainer/train_full_sft.py` 的核心能力：SFT 数据构造、仅监督 assistant、
  支持 MoE aux loss、周期保存 `.pth`、可续训。
- 产物遵循 v2 目录约定，默认写入 `out_v2/`、`checkpoints_v2/`、`logs_v2/`。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import yaml
from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    default_data_collator,
    set_seed,
)
from transformers.modeling_utils import unwrap_model

from trl import SFTConfig, SFTTrainer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_OUT_V2_DIR = os.path.join(_ROOT_DIR, "out_v2")
_CHECKPOINTS_V2_DIR = os.path.join(_ROOT_DIR, "checkpoints_v2")
_LOGS_V2_DIR = os.path.join(_ROOT_DIR, "logs_v2")
_CONFIG_SFT_DIR = os.path.join(_ROOT_DIR, "config", "sft")
_DEFAULT_CONFIG_PATH = os.path.join(_CONFIG_SFT_DIR, "config_v2.yaml")
_DEFAULT_TOKENIZER_PATH = os.path.join(_ROOT_DIR, "model")
_DEFAULT_DATA_PATH = os.path.join(_ROOT_DIR, "dataset", "sft_1024.jsonl")

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from model.model_minimind_v2 import MiniMindConfigV2, MiniMindForCausalLMV2
from trainer.trainer_utils import Logger, get_lr, lm_checkpoint

warnings.filterwarnings("ignore")


def _is_world_process_zero(args) -> bool:
    """与 HF Trainer 一致：分布式仅 rank0 写盘；单卡或未初始化进程组时视为主进程。"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    proc = getattr(args, "process_index", None)
    if proc is not None:
        return int(proc) == 0
    return getattr(args, "local_rank", -1) in (-1, 0)


def resolve_local_path(path: str) -> str:
    path = os.path.expanduser(path.strip())
    if os.path.isabs(path):
        return os.path.normpath(path)
    for base in (_ROOT_DIR, os.getcwd()):
        cand = os.path.normpath(os.path.join(base, path))
        if os.path.isdir(cand) or os.path.isfile(cand):
            return cand
    return os.path.normpath(os.path.join(os.getcwd(), path))


def _load_training_config_defaults(config_path: str) -> dict:
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
    sections = ("training", "distributed", "data", "model", "checkpoint", "logging", "runtime", "monitoring")
    defaults = {}
    for sec in sections:
        block = data.get(sec, {})
        if isinstance(block, dict):
            defaults.update(block)
    for k, v in data.items():
        if not isinstance(v, dict):
            defaults[k] = v
    return defaults


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get("tools") for conv in conversations):
        return conversations
    system_prompts = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [{"role": "system", "content": random.choice(system_prompts)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if "<think>\n\n</think>\n\n" in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


def _ensure_nccl_env_for_rtx4000(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        return
    multi_gpu = int(getattr(args, "num_processes", 1)) > 1
    if not (multi_gpu or int(getattr(args, "use_fsdp", 0)) == 1):
        return
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")


def _strip_launcher_args(argv: list[str]) -> list[str]:
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
    cmd = [
        accelerate_bin,
        "launch",
        "--num_processes",
        str(need),
        os.path.abspath(__file__),
        "--auto_accelerate_launch",
        "0",
    ] + forwarded
    Logger(f"检测到单进程启动，自动切换为 {need} 卡: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=os.environ.copy())
    raise SystemExit(0)


@dataclass
class SFTRecord:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


class SFTDatasetForTRL(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features(
            {
                "conversations": [
                    {
                        "role": Value("string"),
                        "content": Value("string"),
                        "reasoning_content": Value("string"),
                        "tools": Value("string"),
                        "tool_calls": Value("string"),
                    }
                ]
            }
        )
        self.samples = load_dataset("json", data_files=jsonl_path, split="train", features=features)
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def generate_labels(self, input_ids: list[int]) -> list[int]:
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        conversations = pre_processing_chat(sample["conversations"])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[: self.max_length]
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = self.generate_labels(input_ids)
        attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class MiniMindSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is not None and loss is not None:
            loss = loss + aux_loss
        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            optimizer = optimizer if optimizer is not None else self.optimizer

            def lr_lambda(current_step: int):
                if num_training_steps <= 0:
                    return 1.0
                current_lr = get_lr(current_step, num_training_steps, self.args.learning_rate)
                base_lr = max(self.args.learning_rate, 1e-12)
                return float(current_lr / base_lr)

            from torch.optim.lr_scheduler import LambdaLR

            self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler


class SavePthCallback(TrainerCallback):
    def __init__(self, save_dir: str, save_weight: str, lm_config: MiniMindConfigV2, resume_dir: str):
        self.save_dir = save_dir
        self.save_weight = save_weight
        self.lm_config = lm_config
        self.resume_dir = resume_dir

    def _save_weight(self, model) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        moe_suffix = "_moe" if self.lm_config.use_moe else ""
        ckp = f"{self.save_dir}/{self.save_weight}_{self.lm_config.hidden_size}{moe_suffix}.pth"
        raw_model = unwrap_model(model)
        state_dict = raw_model.state_dict()
        torch.save({k: v.detach().half().cpu() for k, v in state_dict.items()}, ckp)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # HF CallbackHandler.call_event 传入 model/optimizer，不会传 trainer
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        if model is None or not _is_world_process_zero(args):
            return control
        self._save_weight(model)
        if optimizer is not None:
            lm_checkpoint(
                self.lm_config,
                weight=self.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=int(state.epoch or 0),
                step=int(state.global_step),
                save_dir=self.resume_dir,
            )
        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        if model is None or not _is_world_process_zero(args):
            return control
        self._save_weight(model)
        if optimizer is not None:
            lm_checkpoint(
                self.lm_config,
                weight=self.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=int(state.epoch or 0),
                step=int(state.global_step),
                save_dir=self.resume_dir,
            )
        return control


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniMind Full SFT v2 (TRL + Transformers)")
    p.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG_PATH,
        help="YAML 配置文件路径（读取 training/data/model/checkpoint/logging/runtime/monitoring 作为默认值）",
    )
    p.add_argument("--save_dir", type=str, default=_OUT_V2_DIR, help="模型保存目录（默认 out_v2）")
    p.add_argument("--save_weight", default="full_sft_v2", type=str, help="保存权重前缀")
    p.add_argument("--epochs", type=int, default=2, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=16, help="每卡 batch size")
    p.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    p.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    p.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    p.add_argument("--log_interval", type=int, default=100, help="日志打印间隔（steps）")
    p.add_argument("--save_interval", type=int, default=1000, help="保存间隔（steps）")
    p.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    p.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    p.add_argument("--max_seq_len", default=768, type=int, help="训练最大长度")
    p.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否启用 MoE")
    p.add_argument("--data_path", type=str, default=_DEFAULT_DATA_PATH, help="SFT jsonl 数据路径")
    p.add_argument("--tokenizer_path", type=str, default=_DEFAULT_TOKENIZER_PATH, help="tokenizer 路径")
    p.add_argument("--from_weight", default="pretrain", type=str, help="初始化权重名；none 表示随机初始化")
    p.add_argument(
        "--from_weight_dir",
        type=str,
        default=None,
        help="初始化权重目录；默认与 --save_dir 相同（可指向旧 out/）",
    )
    p.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否从最新 checkpoint 自动续训")
    p.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    p.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否启用 torch.compile")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--save_total_limit", type=int, default=3, help="HF checkpoint 最多保留数量")
    p.add_argument("--logging_dir", type=str, default=os.path.join(_LOGS_V2_DIR, "full_sft_v2"), help="日志目录")
    p.add_argument("--output_dir", type=str, default=os.path.join(_CHECKPOINTS_V2_DIR, "full_sft_v2"), help="Trainer checkpoint 目录")
    p.add_argument("--use_wandb", action="store_true", help="是否启用 wandb 监控")
    p.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT-v2", help="wandb 项目名")
    p.add_argument("--wandb_run_name", type=str, default="", help="wandb run 名称（空则自动生成）")
    p.add_argument("--use_fsdp", default=0, type=int, choices=[0, 1], help="是否启用 FSDP")
    p.add_argument(
        "--fsdp_sharding",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "no_shard"],
        help="FSDP sharding 策略",
    )
    p.add_argument("--num_processes", type=int, default=1, help="自动 accelerate launch 时的进程数")
    p.add_argument(
        "--auto_accelerate_launch",
        type=int,
        default=1,
        choices=[0, 1],
        help="直接 python 启动且 num_processes>1 时，是否自动切 accelerate launch",
    )
    return p


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    pre_args, _ = parser.parse_known_args()
    cfg_defaults = _load_training_config_defaults(getattr(pre_args, "config", ""))
    if cfg_defaults:
        parser.set_defaults(**cfg_defaults)
    return parser.parse_args()


def find_last_hf_checkpoint(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            step = name.split("checkpoint-")[-1]
            if step.isdigit():
                candidates.append((int(step), os.path.join(output_dir, name)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _list_weight_prefix_candidates(weight_dir: str, hidden_size: int, use_moe: bool) -> list[str]:
    if not os.path.isdir(weight_dir):
        return []
    suffix = f"_{hidden_size}{'_moe' if use_moe else ''}.pth"
    prefixes: list[str] = []
    for name in os.listdir(weight_dir):
        if name.endswith(suffix):
            prefixes.append(name[: -len(suffix)])
    prefixes.sort()
    return prefixes


def main():
    args = parse_args()
    maybe_relaunch_distributed(args)
    _ensure_nccl_env_for_rtx4000(args)
    set_seed(args.seed)

    args.save_dir = resolve_local_path(args.save_dir)
    args.data_path = resolve_local_path(args.data_path)
    args.tokenizer_path = resolve_local_path(args.tokenizer_path)
    args.output_dir = resolve_local_path(args.output_dir)
    args.logging_dir = resolve_local_path(args.logging_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    os.makedirs(_CHECKPOINTS_V2_DIR, exist_ok=True)

    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"SFT 数据不存在: {args.data_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm_config = MiniMindConfigV2(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLMV2(lm_config)

    if args.from_weight != "none":
        moe_suffix = "_moe" if lm_config.use_moe else ""
        from_dir = resolve_local_path(args.from_weight_dir) if args.from_weight_dir else args.save_dir
        from_path = f"{from_dir}/{args.from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        Logger(f"load init weight: {from_path}")
        if not os.path.isfile(from_path):
            candidates = _list_weight_prefix_candidates(from_dir, lm_config.hidden_size, lm_config.use_moe)
            hint = f"可用 from_weight 候选: {candidates}" if candidates else "当前目录下未找到匹配 hidden_size/use_moe 的权重文件。"
            raise FileNotFoundError(
                f"初始化权重不存在: {from_path}\n"
                f"请检查 --from_weight 与 --from_weight_dir。\n{hint}"
            )
        state_dict = torch.load(from_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")

    train_ds = SFTDatasetForTRL(args.data_path, tokenizer, max_length=args.max_seq_len)

    bf16 = args.dtype == "bfloat16"
    fp16 = args.dtype == "float16"
    if getattr(args, "use_wandb", False):
        os.environ["WANDB_PROJECT"] = str(args.wandb_project)
        if getattr(args, "wandb_run_name", ""):
            os.environ["WANDB_NAME"] = str(args.wandb_run_name)
        Logger(f"wandb enabled, project: {args.wandb_project}")

    fsdp = ""
    fsdp_config = None
    if int(args.use_fsdp) == 1:
        fsdp = f"{args.fsdp_sharding} auto_wrap"
        fsdp_config = {"transformer_layer_cls_to_wrap": ["MiniMindBlock"]}

    run_name = getattr(args, "wandb_run_name", "") or (
        f"full-sft-v2-h{args.hidden_size}-l{args.num_hidden_layers}-bs{args.batch_size}-lr{args.learning_rate}"
    )
    training_args = SFTConfig(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        lr_scheduler_type="constant",
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        logging_steps=args.log_interval,
        save_steps=args.save_interval,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        bf16=bf16,
        fp16=fp16,
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        report_to=["wandb"] if getattr(args, "use_wandb", False) else [],
        run_name=run_name,
        remove_unused_columns=False,
        dataset_text_field=None,
    )

    trainer = MiniMindSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )
    trainer.add_callback(SavePthCallback(args.save_dir, args.save_weight, lm_config, _CHECKPOINTS_V2_DIR))

    resume_from = None
    if args.from_resume == 1:
        resume_from = find_last_hf_checkpoint(args.output_dir)
        if resume_from:
            Logger(f"resume from: {resume_from}")
        else:
            ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=_CHECKPOINTS_V2_DIR)
            if ckp_data is not None:
                Logger("found legacy resume file in checkpoints_v2, loading model/optimizer states")
                trainer.model.load_state_dict(ckp_data["model"], strict=False)
                trainer.create_optimizer()
                if "optimizer" in ckp_data and trainer.optimizer is not None:
                    trainer.optimizer.load_state_dict(ckp_data["optimizer"])

    train_result = trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_state()
    metrics: dict[str, Any] = train_result.metrics if train_result is not None else {}
    if trainer.is_world_process_zero():
        Logger(f"train finished, metrics: {metrics}")


if __name__ == "__main__":
    main()
