"""
将 HF Trainer 保存的 checkpoint（含 model.safetensors）转为与 v1 eval_llm 兼容的 .pth。

示例:
  python trainer_v2/export_hf_checkpoint_to_pth_v2.py \\
    --checkpoint_dir checkpoints_v2/full_sft_v2/checkpoint-131138 \\
    --out_path out_v2/full_sft_v2_768.pth
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, required=True, help="含 model.safetensors 的目录")
    p.add_argument("--out_path", type=str, required=True, help="输出的 .pth 路径")
    args = p.parse_args()

    ckpt = os.path.abspath(args.checkpoint_dir)
    st_path = os.path.join(ckpt, "model.safetensors")
    bin_path = os.path.join(ckpt, "pytorch_model.bin")
    if os.path.isfile(st_path):
        from safetensors.torch import load_file

        state_dict = load_file(st_path)
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"未找到 {st_path} 或 {bin_path}")

    out = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({k: v.detach().half().cpu() for k, v in state_dict.items()}, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
