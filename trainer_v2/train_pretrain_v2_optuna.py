"""
使用 Optuna 对 train_pretrain_v2 做超参搜索（产物在 out_v2/optuna_runs/ 与 checkpoints_v2 下独立 save_weight）。

依赖：已安装 optuna（见 requirements.txt）。

示例（建议先小步数搜索）:
  cd /path/to/minimind
  python trainer_v2/train_pretrain_v2_optuna.py --n_trials 10 --max_train_steps 600 --epochs 1

可视化（可选）:
  optuna-dashboard sqlite:///optuna_v2/optuna.db
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_OPTUNA_DIR = os.path.join(_ROOT_DIR, "optuna_v2")
_OUT_V2_DIR = os.path.join(_ROOT_DIR, "out_v2")
_LOGS_V2_DIR = os.path.join(_ROOT_DIR, "logs_v2")
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import optuna
from optuna.pruners import MedianPruner

from trainer_v2.train_pretrain_v2 import build_parser


def _trial_namespace(base: argparse.Namespace, trial: optuna.Trial) -> argparse.Namespace:
    a = argparse.Namespace(**vars(base))
    a.from_resume = 0
    a.use_wandb = False
    a.no_save_checkpoint = True
    a.max_optimizer_steps = getattr(base, "max_optimizer_steps", 0)
    a.target_tokens = getattr(base, "target_tokens", 0)
    a.stage = getattr(base, "stage", "optuna_pretrain_v2")
    a.logs_dir = getattr(base, "logs_dir", _LOGS_V2_DIR)
    a.log_to_file = getattr(base, "log_to_file", 1)
    a.log_to_terminal = getattr(base, "log_to_terminal", 0)

    a.save_dir = os.path.join(_OUT_V2_DIR, "optuna_runs", f"trial_{trial.number}")
    a.save_weight = f"{base.save_weight}_opt{trial.number}"

    a.learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    a.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    a.accumulation_steps = trial.suggest_int("accumulation_steps", 4, 32, step=4)
    a.grad_clip = trial.suggest_float("grad_clip", 0.5, 2.0)
    a.weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    if base.max_optimizer_steps > 0:
        # 优先按真实 optimizer 更新步数截断；避免再被 max_train_steps 间接影响
        a.max_optimizer_steps = base.max_optimizer_steps
        a.max_train_steps = 0
    elif base.target_tokens > 0:
        # 其次按 token 数截断，避免与 max_train_steps 重复生效
        a.target_tokens = base.target_tokens
        a.max_train_steps = 0
    else:
        if base.max_train_steps <= 0:
            a.max_train_steps = 3200
        else:
            a.max_train_steps = base.max_train_steps

    a.log_interval = min(base.log_interval, 50)
    return a


def objective(trial: optuna.Trial, base: argparse.Namespace) -> float:
    a = _trial_namespace(base, trial)
    os.makedirs(a.save_dir, exist_ok=True)
    trial_log_dir = os.path.join(a.logs_dir, a.stage, a.run_tag if getattr(a, "run_tag", "") else "optuna-run", f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)

    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(a.num_processes),
        os.path.join(_ROOT_DIR, "trainer_v2", "train_pretrain_v2.py"),
        "--auto_accelerate_launch",
        "0",
        "--num_processes",
        str(a.num_processes),
        "--save_dir",
        a.save_dir,
        "--save_weight",
        a.save_weight,
        "--epochs",
        str(a.epochs),
        "--batch_size",
        str(a.batch_size),
        "--learning_rate",
        str(a.learning_rate),
        "--weight_decay",
        str(a.weight_decay),
        "--dtype",
        str(a.dtype),
        "--num_workers",
        str(a.num_workers),
        "--accumulation_steps",
        str(a.accumulation_steps),
        "--grad_clip",
        str(a.grad_clip),
        "--log_interval",
        str(a.log_interval),
        "--save_interval",
        str(a.save_interval),
        "--hidden_size",
        str(a.hidden_size),
        "--num_hidden_layers",
        str(a.num_hidden_layers),
        "--max_seq_len",
        str(a.max_seq_len),
        "--use_moe",
        str(a.use_moe),
        "--data_path",
        str(a.data_path),
        "--from_weight",
        str(a.from_weight),
        "--from_resume",
        str(a.from_resume),
        "--tokenizer_path",
        str(a.tokenizer_path),
        "--use_compile",
        str(a.use_compile),
        "--use_fsdp",
        str(a.use_fsdp),
        "--fsdp_sharding",
        str(a.fsdp_sharding),
        "--max_train_steps",
        str(a.max_train_steps),
        "--max_optimizer_steps",
        str(a.max_optimizer_steps),
        "--target_tokens",
        str(a.target_tokens),
        "--stage",
        str(a.stage),
        "--run_tag",
        str(getattr(a, "run_tag", "")),
        "--logs_dir",
        str(a.logs_dir),
        "--log_to_file",
        str(a.log_to_file),
        "--log_to_terminal",
        str(a.log_to_terminal),
    ]
    if a.from_weight_dir:
        cmd += ["--from_weight_dir", str(a.from_weight_dir)]

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["MM_LAUNCH_DIR"] = trial_log_dir
        p = subprocess.Popen(
            cmd,
            cwd=_ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        lines: list[str] = []
        last_output = time.time()
        started_at = time.time()
        runner_log = open(os.path.join(trial_log_dir, "trial_runner.log"), "a", encoding="utf-8", buffering=1)
        assert p.stdout is not None
        while True:
            line = p.stdout.readline()
            if line:
                if int(getattr(base, "subprocess_log_to_terminal", 0)) == 1:
                    print(line, end="")
                runner_log.write(line)
                lines.append(line)
                last_output = time.time()
            elif p.poll() is not None:
                break
            elif time.time() - last_output > base.subprocess_heartbeat_sec:
                elapsed = int(time.time() - started_at)
                hb = (
                    f"[Optuna][Trial {trial.number}] 子进程 {base.subprocess_heartbeat_sec}s 无新日志，"
                    f"仍在运行中（elapsed={elapsed}s）。可用 nvidia-smi 观察 GPU 利用率。\n"
                )
                if int(getattr(base, "subprocess_log_to_terminal", 0)) == 1:
                    print(hb, end="")
                runner_log.write(hb)
                last_output = time.time()
            else:
                time.sleep(0.2)

        rc = p.wait()
        runner_log.close()
        text = "".join(lines)
        if rc != 0:
            raise RuntimeError(f"trial 子进程退出码非 0: {rc}")
        # 仅匹配训练主日志行上的主损失，避免误抓到 aux_loss: 0.0000
        # 例如: Epoch:[1/1](10/11040), loss: 8.4114, logits_loss: ...
        matches = re.findall(
            r"Epoch:\[[^\]]+\]\([^\)]*\),\s*loss:\s*([0-9]+(?:\.[0-9]+)?)",
            text,
        )
        if not matches:
            raise RuntimeError("未能从训练日志中解析出 loss，请检查训练输出或减小 log_interval。")
        loss = float(matches[-1])
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return loss


def extend_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--n_trials", type=int, default=20, help="Optuna 试验次数")
    p.add_argument(
        "--optuna_storage",
        type=str,
        default=f"sqlite:///{os.path.join(_OPTUNA_DIR, 'optuna.db')}",
        help="Optuna storage URL（默认项目根下 optuna_v2/optuna.db）",
    )
    p.add_argument("--study_name", type=str, default="minimind_pretrain_v2")
    p.add_argument("--optuna_seed", type=int, default=42)
    p.add_argument(
        "--subprocess_heartbeat_sec",
        type=int,
        default=180,
        help="子进程超过该秒数无日志时打印心跳提示（便于排查“看起来卡住”）",
    )
    p.add_argument(
        "--subprocess_log_to_terminal",
        type=int,
        default=0,
        choices=[0, 1],
        help="是否将每个 trial 子进程日志同时打印到终端（默认仅写入 logs_v2）",
    )
    p.set_defaults(stage="optuna_pretrain_v2", log_to_file=1, log_to_terminal=0)
    return p


def main():
    p = build_parser()
    extend_parser(p)
    args = p.parse_args()
    if not getattr(args, "run_tag", ""):
        args.run_tag = f"optuna-{time.strftime('%Y%m%d-%H%M%S')}"

    os.makedirs(_OPTUNA_DIR, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.optuna_storage,
        load_if_exists=True,
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=args.optuna_seed),
    )

    if args.max_optimizer_steps > 0:
        print(
            f"[Optuna] 使用 max_optimizer_steps={args.max_optimizer_steps} 控制每个 trial 的真实 optimizer 更新步数。"
        )
    elif args.target_tokens > 0:
        print(
            f"[Optuna] 使用 target_tokens={args.target_tokens} 控制每个 trial 的全局 token 数。"
        )
    elif args.max_train_steps <= 0:
        print(
            "[Optuna] 提示: 未设置 --max_train_steps 时，每个 trial 内会使用默认 3200 步（在 trial 内覆盖）；"
            "也可显式传入 --max_train_steps 控制搜索成本。"
        )

    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials, show_progress_bar=True)

    print("=== Optuna 完成 ===")
    print("best_trial:", study.best_trial.number)
    print("best_value:", study.best_value)
    print("best_params:", study.best_params)


if __name__ == "__main__":
    main()
