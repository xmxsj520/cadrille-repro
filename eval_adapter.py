#!/usr/bin/env python3
"""
统一评估适配器 —— 把任意 baseline 的预测，转成 cadrille 的评估口径
=====================================================================
用途：Table 1 全表复现的地基。

核心原则
--------
**所有 baseline 必须过同一套指标计算。** 直接抄别人论文里的数字填进
Table 1 是不成立的，因为各家的归一化范围、CD 的 mean/median、
点采样数、IR 判定标准都不一样。

指标函数是从官方 `evaluate.py` 逐行对齐过来的，改动只有两处（见下）。
不要为了让数字好看去改这些函数。

与官方 evaluate.py 的差异（仅此两处）
-------------------------------------
1. 官方只接受 CadQuery `.py` 文件；本脚本通过「预测加载器」抽象，
   同时支持 `.py`(CadQuery 代码) / `.stl` / `.obj` / `.ply`(mesh 文件)。
2. 官方把 CD 乘 1000 只是打印时的单位换算；本脚本存原始值，导出时再换算。

指标口径（论文 §5 Metrics）
--------------------------
- GT 归一化到 [-0.5, 0.5]^3
- CD: 8192 点表面采样, 双向平方距离均值; 跨样本取 median, 再 ×1000
- IoU: 布尔交集体积 / 并集体积 (逐 body 配对求和)
- IR: 无法产出合法 mesh 的预测占比
- Table 1 口径: 每个输入只取 1 个预测 (n_samples=1)

用法
----
# 1) 官方格式: work_dirs/tmp_py 里是 name+0.py
python eval_adapter.py --gt-mesh-path ./data/deepcad_test_mesh \
                       --pred-dir ./work_dirs/tmp_py --loader cq_py \
                       --out ./results/cadrille_R_pi_pc.json

# 2) baseline 直接输出 mesh: pred 目录里是 name.stl
python eval_adapter.py --gt-mesh-path ./data/deepcad_test_mesh \
                       --pred-dir ./preds/cadsignet --loader mesh \
                       --out ./results/cadsignet_pc.json

# 3) 多种采样(测 IR@k / IoU@k): 预测名形如 name+0.stl, name+1.stl ...
python eval_adapter.py ... --loader mesh --n-samples 10

# 4) 只看前 200 个样本(快速验证链路)
python eval_adapter.py ... --limit 200
"""

import os
import re
import sys
import json
import glob
import argparse
import traceback
from collections import defaultdict
from multiprocessing import Process
from multiprocessing.pool import Pool

import numpy as np
import trimesh
from tqdm import tqdm
from scipy.spatial import cKDTree


# ============================================================================
# 指标计算 —— 与官方 evaluate.py 逐行对齐，不要改
# ============================================================================

def compute_chamfer_distance(gt_mesh, pred_mesh, n_points):
    """双向平方 Chamfer Distance 的均值。官方实现原样复制。"""
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    return np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))


def compute_iou(gt_mesh, pred_mesh):
    """布尔交集体积 / 并集体积。官方实现原样复制(含逐 body 配对)。"""
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0
                intersection_volume += volume

        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return intersection_volume / union_volume
    except Exception:
        return None


def normalize_pred_mesh(pred_mesh):
    """预测 mesh 居中 -> 缩放到单位立方体 -> 平移到 [0,1]^3。官方实现原样复制。"""
    center = (pred_mesh.bounds[0] + pred_mesh.bounds[1]) / 2.0
    pred_mesh.apply_translation(-center)
    extent = np.max(pred_mesh.extents)
    if extent > 1e-7:
        pred_mesh.apply_scale(1.0 / extent)
    pred_mesh.apply_transform(
        trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return pred_mesh


# ============================================================================
# 预测加载器 —— 这一层是「适配 baseline」的唯一入口
# ============================================================================
# 每种 baseline 的输出格式不同，加新方法时**只在这里加一个 loader**，
# 不要去动上面的指标函数。

def load_cq_py(path, timeout=3.0):
    """CadQuery Python 代码 -> mesh。官方路径: exec 后取 globals()['r']。

    ⚠️ 官方用 3 秒超时的子进程, 理由是 CadQuery 有内存泄漏。
    ⚠️ 若你的机器 CPU 较弱, 3 秒会把本该成功的样本判为 invalid, 导致 IR 虚高。
       租独占机器, 或显式调大 timeout 并在报告里注明。
    """
    res = {}

    def _work():
        try:
            import cadquery as cq  # noqa: F401  (代码里 import cadquery as cq)
            with open(path, 'r') as f:
                code = f.read()
            g = {}
            exec(code, g)
            compound = g['r'].val()
            verts, faces = compound.tessellate(0.001, 0.1)
            res['mesh'] = trimesh.Trimesh(
                [(v.x, v.y, v.z) for v in verts], faces)
        except Exception as exc:
            res['err'] = f'{type(exc).__name__}: {exc}'

    p = Process(target=_work)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        res['err'] = f'timeout>{timeout}s'
    return res.get('mesh'), res.get('err')


def load_mesh_file(path, timeout=30.0):
    """直接读 mesh 文件(.stl/.obj/.ply)。用于输出 mesh 的 baseline。"""
    try:
        mesh = trimesh.load_mesh(path)
        if len(mesh.faces) <= 2:
            return None, 'faces<=2'
        return mesh, None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


LOADERS = {
    'cq_py': load_cq_py,      # CadQuery 代码
    'mesh': load_mesh_file,   # 已经是 mesh 文件
}


# ============================================================================
# 文件枚举 —— 处理 "name+0.py" / "name.stl" 两种命名
# ============================================================================

def scan_predictions(pred_dir, loader):
    """返回 {sample_name: [路径, ...]}，按 sample 聚合多次采样。

    支持两种命名:
      - 官方风格: <name>+<k>.py      (k = 采样序号)
      - 普通风格: <name>.stl / <name>.py   (视为 k=0)
    """
    ext = '.py' if loader == 'cq_py' else None
    files = [f for f in os.listdir(pred_dir)
             if (f.endswith(ext) if ext else
                 f.lower().endswith(('.stl', '.obj', '.ply')))]
    groups = defaultdict(list)
    for f in sorted(files):
        m = re.match(r'^(?P<name>.+?)\+(?P<idx>\d+)\.[A-Za-z0-9]+$', f)
        if m:
            groups[m.group('name')].append((int(m.group('idx')), f))
        else:
            stem = os.path.splitext(f)[0]
            groups[stem].append((0, f))
    return {k: [f for _, f in sorted(v)] for k, v in groups.items()}


# ============================================================================
# 单样本评估
# ============================================================================

def eval_one(name, paths, gt_dir, pred_dir, loader, n_points, timeout):
    fn = LOADERS[loader]
    gt_path = os.path.join(gt_dir, name + '.stl')
    if not os.path.exists(gt_path):
        return dict(name=name, cd=None, iou=None, invalid=1,
                    err=f'GT 缺失: {gt_path}')

    gt_mesh = trimesh.load_mesh(gt_path)

    cds, ious, errs = [], [], []
    for p in paths:
        mesh, err = fn(os.path.join(pred_dir, p))
        if mesh is None:
            errs.append(f'{p}: {err}')
            continue
        try:
            mesh = normalize_pred_mesh(mesh)
            cd = compute_chamfer_distance(gt_mesh, mesh, n_points)
            iou = compute_iou(gt_mesh, mesh)
            cds.append(cd)
            if iou is not None:
                ious.append(iou)
        except Exception as exc:
            errs.append(f'{p}: {type(exc).__name__}: {exc}')

    if not cds:
        return dict(name=name, cd=None, iou=None, invalid=1,
                    err='; '.join(errs[:3]) if errs else 'no valid sample')

    # Table 1 口径: 每样本取 CD 最小(官方做法) 与 IoU 最大
    return dict(name=name, cd=float(min(cds)),
                iou=float(max(ious)) if ious else None,
                invalid=0, err=None)


def _worker(args):
    try:
        return eval_one(*args)
    except Exception:
        return dict(name=args[0], cd=None, iou=None, invalid=1,
                    err=traceback.format_exc(limit=2))


# ============================================================================
# 汇总
# ============================================================================

def summarize(records, n_samples, label):
    total = len(records)
    valid = [r for r in records if r['cd'] is not None and r['invalid'] == 0]
    cd = sorted(r['cd'] for r in valid)
    iou = [r['iou'] for r in valid if r['iou'] is not None]
    ir = total - len(valid)

    out = {
        'label': label,
        'n_samples': n_samples,
        'n_total': total,
        'n_valid': len(valid),
        # 论文口径: median CD, 单位 ×10^3
        'median_cd_x1e3': round(float(np.median(cd)) * 1000, 3) if cd else None,
        'mean_cd_x1e3': round(float(np.mean(cd)) * 1000, 3) if cd else None,
        # 论文报的就是 mean IoU
        'mean_iou': round(float(np.mean(iou)) * 100, 3) if iou else None,
        'ir_percent': round(ir / total * 100, 2) if total else None,
    }
    return out, records


def main():
    ap = argparse.ArgumentParser(description='cadrille 统一评估适配器')
    ap.add_argument('--gt-mesh-path', required=True,
                    help='GT stl 目录, 如 ./data/deepcad_test_mesh')
    ap.add_argument('--pred-dir', required=True,
                    help='预测目录, 含 <name>+<k>.py 或 <name>.stl')
    ap.add_argument('--loader', default='cq_py', choices=sorted(LOADERS.keys()),
                    help='预测格式: cq_py(CadQuery 代码) / mesh(mesh 文件)')
    ap.add_argument('--out', default=None, help='结果 JSON 输出路径')
    ap.add_argument('--n-points', type=int, default=8192,
                    help='CD 采样点数, 官方 8192')
    ap.add_argument('--n-samples', type=int, default=1,
                    help='Table 1 口径 = 1; 测 appendix 表 11 用 10')
    ap.add_argument('--timeout', type=float, default=3.0,
                    help='cq_py 单样本执行超时(秒), 官方 3.0')
    ap.add_argument('--workers', type=int, default=16, help='并行进程数')
    ap.add_argument('--limit', type=int, default=None, help='只评前 N 个样本')
    ap.add_argument('--label', default=None, help='结果里的方法名标注')
    args = ap.parse_args()

    groups = scan_predictions(args.pred_dir, args.loader)
    if not groups:
        print(f'[FAIL] {args.pred_dir} 里没找到可评估的预测文件 '
              f'(loader={args.loader})')
        sys.exit(1)

    names = sorted(groups.keys())
    if args.limit:
        names = names[:args.limit]

    # 只保留前 n_samples 个采样
    tasks = [(n, groups[n][:args.n_samples], args.gt_mesh_path, args.pred_dir,
              args.loader, args.n_points, args.timeout) for n in names]

    print(f'评估中: {len(tasks)} 个样本, loader={args.loader}, '
          f'n_points={args.n_points}, n_samples={args.n_samples}, '
          f'workers={args.workers}')
    if args.loader == 'cq_py':
        print(f'⚠️ CadQuery 执行超时 = {args.timeout}s。'
              f'机器 CPU 弱会让 IR 虚高, 请在报告中注明实际值。')

    records = []
    with Pool(args.workers) as pool:
        for rec in tqdm(pool.imap(_worker, tasks), total=len(tasks)):
            records.append(rec)

    label = args.label or os.path.basename(os.path.normpath(args.pred_dir))
    summary, records = summarize(records, args.n_samples, label)

    print('\n' + '=' * 64)
    print(f'方法: {summary["label"]}')
    print(f'样本: {summary["n_valid"]}/{summary["n_total"]} 有效')
    print(f'median CD (×10^3): {summary["median_cd_x1e3"]}')
    print(f'mean   CD (×10^3): {summary["mean_cd_x1e3"]}')
    print(f'mean IoU (%):      {summary["mean_iou"]}')
    print(f'IR (%):            {summary["ir_percent"]}')
    print('=' * 64)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'summary': summary, 'per_sample': records}, f, indent=2)
        print(f'已写入 {args.out}')

    # 失败原因归类, 方便排查是"方法不行"还是"你的 loader 写错了"
    errs = defaultdict(int)
    for r in records:
        if r['invalid']:
            key = (r['err'] or 'unknown')[:80]
            errs[key] += 1
    if errs:
        print('\n无效样本原因 Top 5:')
        for k, v in sorted(errs.items(), key=lambda kv: -kv[1])[:5]:
            print(f'  {v:5d}  {k}')


if __name__ == '__main__':
    sys.exit(main())
