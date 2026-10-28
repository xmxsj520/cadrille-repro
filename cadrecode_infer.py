#!/usr/bin/env python3
"""
CAD-Recode v1.5 批量推理脚本（复现 Table 1 的 CAD-Recode R_p 行）
=====================================================================
基于官方 demo.ipynb 改写，用于在 DeepCAD 测试集（8046 个 stl）上批量生成 CadQuery 代码。

与官方 demo 的唯一差异：
  - attn_implementation 用 'sdpa'（本机未安装 flash-attn）
  - 支持批量推理
  - 输出文件名格式 <name>+0.py，与 cadrille 的 evaluate.py 兼容

用法：
  python cadrecode_infer.py --data-dir ./data/deepcad_test_mesh \
                            --out-dir ./work_dirs/tmp_py_cadrecode \
                            --ckpt ./ckpt_cadrecode --tokenizer ./ckpt_qwen2 \
                            --batch-size 16

  # 先跑 20 个验证链路
  python cadrecode_infer.py ... --limit 20
"""

import os
import sys
import argparse

import numpy as np
import torch
import trimesh
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Model, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from pytorch3d.ops import sample_farthest_points

N_POINTS = 256
MAX_NEW_TOKENS = 768


# ============================================================================
# 模型定义（与官方 demo.ipynb 完全一致）
# ============================================================================

class FourierPointEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points):
        x = points
        x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
        x = torch.cat((points, x.sin(), x.cos()), dim=-1)
        x = self.projection(x)
        return x


class CADRecode(Qwen2ForCausalLM):
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

    def forward(self, input_ids=None, attention_mask=None, point_cloud=None,
                position_ids=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, cache_position=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 把点云 embedding 塞进 attention_mask == -1 的位置
        if past_key_values is None or past_key_values.get_seq_length() == 0:
            assert inputs_embeds is None
            inputs_embeds = self.model.embed_tokens(input_ids)
            point_embeds = self.point_encoder(point_cloud).bfloat16()
            inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
            attention_mask[attention_mask == -1] = 1
            input_ids = None
            position_ids = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs["point_cloud"] = kwargs["point_cloud"]
        return model_inputs


# ============================================================================
# 点云采样（与官方 demo 一致：居中 + 缩放到最大边长 = 2，再 FPS 256 点）
# ============================================================================

def mesh_to_point_cloud(path, n_points=N_POINTS, n_pre_points=8192):
    mesh = trimesh.load_mesh(path)
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    mesh.apply_scale(2.0 / max(mesh.extents))
    vertices, _ = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(
        torch.tensor(vertices, dtype=torch.float32).unsqueeze(0), K=n_points)
    return np.asarray(vertices[ids.numpy()[0]])


# ============================================================================
# 从生成的文本中抽取代码
# ============================================================================

def extract_code(py_string):
    begin = py_string.find("<|im_start|>")
    if begin >= 0:
        begin += len("<|im_start|>")
    else:
        begin = 0
    end = py_string.find("<|endoftext|>")
    if end < 0:
        end = len(py_string)
    code = py_string[begin:end]
    # 去掉可能出现的 markdown 代码围栏
    code = code.replace("```python", "").replace("```", "")
    return code.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/deepcad_test_mesh")
    ap.add_argument("--out-dir", default="./work_dirs/tmp_py_cadrecode")
    ap.add_argument("--ckpt", default="./ckpt_cadrecode")
    ap.add_argument("--tokenizer", default="./ckpt_qwen2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if len(os.listdir(args.out_dir)) > 0:
        print(f"[FAIL] 输出目录非空: {args.out_dir}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, pad_token="<|im_end|>", padding_side="left")
    model = CADRecode.from_pretrained(
        args.ckpt, torch_dtype="auto", attn_implementation="sdpa"
    ).eval().to(device)
    print("model loaded")

    stems = sorted(f[:-4] for f in os.listdir(args.data_dir) if f.endswith(".stl"))
    if args.limit:
        stems = stems[: args.limit]
    print(f"samples: {len(stems)}")

    im_start_id = tokenizer("<|im_start|>")["input_ids"][0]
    prefix_ids = [tokenizer.pad_token_id] * N_POINTS + [im_start_id]

    for i in tqdm(range(0, len(stems), args.batch_size), ncols=100):
        batch_stems = stems[i:i + args.batch_size]
        pcs, valid_stems = [], []
        for s in batch_stems:
            try:
                pcs.append(mesh_to_point_cloud(os.path.join(args.data_dir, s + ".stl")))
                valid_stems.append(s)
            except Exception as exc:
                print(f"[WARN] 采样失败 {s}: {exc}")
        if not pcs:
            continue

        pc = torch.tensor(np.stack(pcs), dtype=torch.float32).to(device)
        input_ids = torch.tensor([prefix_ids] * len(pcs), device=device)
        attention_mask = torch.tensor(
            [[-1] * N_POINTS + [1]] * len(pcs), device=device)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                point_cloud=pc,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id)

        texts = tokenizer.batch_decode(out, skip_special_tokens=False)
        for s, t in zip(valid_stems, texts):
            with open(os.path.join(args.out_dir, f"{s}+0.py"), "w") as f:
                f.write(extract_code(t))

    print(f"done: {len(os.listdir(args.out_dir))} files -> {args.out_dir}")


if __name__ == "__main__":
    main()
