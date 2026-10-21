#!/usr/bin/env python3
"""
cadrille 复现 - 第一步环境自检
================================
用途：在租来的 GPU 机器上，下数据集之前跑一遍，确认这套代码的依赖真的能跑。
其中第 7 项(Open3D 无头渲染)和第 8 项(CadQuery 执行+网格化)是最容易崩、
且只有真机才暴露的问题，务必看到 [OK]。

用法:
    python step1_check_env.py
    python step1_check_env.py --data-dir /root/autodl-tmp/cadrille/data
    python step1_check_env.py --skip-render      # 只想快点看 import

退出码: 0 = 全部关键项通过; 1 = 有 FAIL
"""

import os
import sys
import json
import time
import shutil
import argparse
import importlib
import traceback
import urllib.request

RESULT = []          # (name, level, detail)   level: OK / WARN / FAIL / SKIP


def record(name, level, detail=""):
    RESULT.append((name, level, detail))
    tag = {"OK": "[OK]  ", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[level]
    print(f"{tag} {name}" + (f"  -> {detail}" if detail else ""))
    sys.stdout.flush()


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    sys.stdout.flush()


def try_import(module_name, attr=None, pip_hint=None, critical=True):
    """导入一个模块(可选再取一个属性)，失败时记录并给出修复提示。"""
    label = module_name + (f".{attr}" if attr else "")
    try:
        mod = importlib.import_module(module_name)
        if attr:
            getattr(mod, attr)
        ver = getattr(mod, "__version__", None)
        record(f"import {label}", "OK", f"version={ver}" if ver else "no __version__")
        return mod
    except Exception as exc:
        level = "FAIL" if critical else "WARN"
        hint = f" 修复: pip install {pip_hint}" if pip_hint else ""
        record(f"import {label}", level, f"{type(exc).__name__}: {exc}{hint}")
        return None


# ----------------------------------------------------------------------------
# 1. Python / 平台
# ----------------------------------------------------------------------------
def check_python():
    section("1. Python 与系统平台")
    v = sys.version_info
    if v.major == 3 and v.minor == 11:
        record("Python 版本", "OK", f"{v.major}.{v.minor}.{v.micro} (官方环境为 3.11)")
    elif v.major == 3 and 10 <= v.minor <= 12:
        record("Python 版本", "WARN", f"{v.major}.{v.minor} 非官方 3.11, 可能出现依赖不兼容")
    else:
        record("Python 版本", "FAIL", f"{v.major}.{v.minor} 不在支持范围")

    record("平台", "OK", f"{sys.platform} / {os.uname().machine if hasattr(os, 'uname') else '?'}")

    # 关键的编译期环境变量，pytorch3d / flash-attn 编译需要
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch:
        record("TORCH_CUDA_ARCH_LIST", "OK", arch)
    else:
        record("TORCH_CUDA_ARCH_LIST", "WARN",
               "未设置; 编译 pytorch3d 前建议设为 8.0(A100)/9.0(H100)")


# ----------------------------------------------------------------------------
# 2. GPU 与 CUDA —— 选卡是否选错的判定就在这里
# ----------------------------------------------------------------------------
def check_gpu():
    section("2. GPU / CUDA / bf16")
    torch = try_import("torch", pip_hint="torch==2.5.1 (--index-url .../cu124)")
    if torch is None:
        record("GPU 检查", "SKIP", "torch 不可用")
        return None

    record("torch 版本", "OK", torch.__version__)
    record("torch 编译的 CUDA", "OK" if torch.version.cuda == "12.4" else "WARN",
           f"{torch.version.cuda} (官方 Dockerfile 为 12.4)")

    if not torch.cuda.is_available():
        record("CUDA 可用性", "FAIL", "torch.cuda.is_available() = False, 检查驱动/是否是 CPU 实例")
        return torch

    record("CUDA 可用性", "OK", f"device_count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cc = torch.cuda.get_device_capability(i)
        props = torch.cuda.get_device_properties(i)
        vram = props.total_memory / 1024 ** 3
        detail = f"{name} | sm_{cc[0]}{cc[1]} | {vram:.1f} GB"
        level = "OK"
        problem = None

        # torch 2.5.1 不支持 sm_120(50 系)，这是最常见的选卡错误
        if cc[0] >= 12:
            level, problem = "FAIL", "50 系(sm_120)不受 torch 2.5.1 支持, pytorch3d/flash-attn 编不过"

        # 检查该架构是否真的被编译进 torch
        try:
            supported = torch.cuda.get_arch_list()
            if supported and f"sm_{cc[0]}{cc[1]}" not in supported:
                level = "FAIL"
                problem = f"torch 未包含 sm_{cc[0]}{cc[1]}, 已编译架构={supported}"
        except Exception:
            pass

        if cc[0] < 8:
            level, problem = "FAIL", "低于 sm_80 不支持 bf16"
        record(f"GPU {i}", level, detail + (f"  <<< {problem}" if problem else ""))

    if not torch.cuda.is_bf16_supported():
        record("bf16 支持", "FAIL", "代码里 torch_dtype=torch.bfloat16 会出问题")
    else:
        record("bf16 支持", "OK", "可用")

    try:
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        b = (a @ a).float().mean().item()
        record("bf16 实算", "OK", f"matmul mean={b:.4f}")
    except Exception as exc:
        record("bf16 实算", "FAIL", f"{type(exc).__name__}: {exc}")

    return torch


# ----------------------------------------------------------------------------
# 3. 依赖 import 全表
# ----------------------------------------------------------------------------
def check_imports():
    section("3. 依赖 import 检查")
    try_import("numpy", pip_hint="numpy==2.2.0")
    try_import("scipy", pip_hint="scipy==1.14.1")
    try_import("trimesh", pip_hint="trimesh==4.5.3")
    try_import("skimage", pip_hint="scikit-image==0.25.0")
    try_import("PIL", pip_hint="Pillow")
    try_import("open3d", pip_hint="需源码编译 headless 版本(见 setup 说明)")
    try_import("pytorch3d", pip_hint="git+.../pytorch3d@06a76ef")
    try_import("pytorch3d.ops", attr="sample_farthest_points",
               pip_hint="pytorch3d 编译不完整")
    try_import("cadquery", pip_hint="git+.../cadquery@e99a15d")
    try_import("OCP", pip_hint="cadquery-ocp==7.7.2", critical=False)
    try_import("ezdxf", pip_hint="ezdxf==1.3.5", critical=False)
    try_import("typish", pip_hint="typish==1.9.3", critical=False)
    try_import("multimethod", pip_hint="multimethod==1.12", critical=False)
    try_import("transformers", pip_hint="transformers==4.50.3")
    try_import("tokenizers", pip_hint="tokenizers==0.21.0", critical=False)
    try_import("accelerate", pip_hint="accelerate==0.34.2")
    try_import("qwen_vl_utils", attr="process_vision_info", pip_hint="qwen-vl-utils==0.0.10")
    try_import("flash_attn", pip_hint="flash-attn==2.7.2.post1")

    # pytorch3d 的 FPS 实算(CPU 即可)
    try:
        import torch
        from pytorch3d.ops import sample_farthest_points
        pts = torch.rand(1, 8192, 3)
        _, ids = sample_farthest_points(pts, K=256)
        assert ids.shape == (1, 256), ids.shape
        record("pytorch3d FPS 实算", "OK", f"ids.shape={tuple(ids.shape)}")
    except Exception as exc:
        record("pytorch3d FPS 实算", "FAIL", f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------
# 4. 关键版本对齐(论文/Dockerfile 钉死的版本)
# ----------------------------------------------------------------------------
def check_versions():
    section("4. 关键版本对齐 (官方 Dockerfile 钉死的版本)")
    expected = {
        "numpy": "2.2.0", "scipy": "1.14.1", "trimesh": "4.5.3",
        "scikit-image": "0.25.0", "transformers": "4.50.3",
        "tokenizers": "0.21.0", "accelerate": "0.34.2",
        "safetensors": "0.4.5", "huggingface-hub": "0.27.0",
        "einops": "0.8.0", "qwen-vl-utils": "0.0.10",
        "cadquery-ocp": "7.7.2", "flash-attn": "2.7.2.post1",
    }
    try:
        from importlib.metadata import version, PackageNotFoundError
    except Exception:
        record("版本对齐", "SKIP", "importlib.metadata 不可用")
        return
    for pkg, want in expected.items():
        try:
            got = version(pkg)
            if got == want:
                record(f"{pkg}", "OK", got)
            else:
                record(f"{pkg}", "WARN", f"实际 {got}, 官方 {want}")
        except PackageNotFoundError:
            record(f"{pkg}", "WARN", f"未安装(官方 {want})")
        except Exception as exc:
            record(f"{pkg}", "WARN", f"查询失败: {exc}")


# ----------------------------------------------------------------------------
# 5. 磁盘 / CPU —— 图像模态的瓶颈在 CPU，不在 GPU
# ----------------------------------------------------------------------------
def check_resources(data_dir):
    section("5. 资源 (磁盘 / CPU核数)")
    for path in [data_dir, os.getcwd(), "/tmp"]:
        try:
            os.makedirs(path, exist_ok=True)
            total, used, free = shutil.disk_usage(path)
            free_gb = free / 1024 ** 3
            level = "OK" if free_gb > 60 else ("WARN" if free_gb > 20 else "FAIL")
            record(f"磁盘 {path}", level,
                   f"可用 {free_gb:.1f} GB / 共 {total / 1024 ** 3:.1f} GB"
                   + ("  建议 >=60GB(数据集+checkpoint)" if free_gb <= 60 else ""))
        except Exception as exc:
            record(f"磁盘 {path}", "WARN", str(exc))

    cores = os.cpu_count() or 0
    level = "OK" if cores >= 16 else "WARN"
    record("CPU 核数", level,
           f"{cores} 核" + ("  train.py 默认 dataloader_num_workers=18, 核少会成为瓶颈" if cores < 16 else ""))

    try:
        import psutil
        mem = psutil.virtual_memory().total / 1024 ** 3
        record("内存", "OK" if mem >= 60 else "WARN",
               f"{mem:.1f} GB" + ("  编译 flash-attn 峰值需 20GB+" if mem < 60 else ""))
    except Exception:
        record("内存", "SKIP", "未安装 psutil")


# ----------------------------------------------------------------------------
# 6. 网络: 能不能拉到 HF 上的数据和权重
# ----------------------------------------------------------------------------
def check_network():
    section("6. 网络连通性 (HuggingFace)")
    urls = {
        "huggingface.co": "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/raw/main/config.json",
        "hf-mirror.com(国内镜像)": "https://hf-mirror.com/Qwen/Qwen2-VL-2B-Instruct/raw/main/config.json",
    }
    any_ok = False
    for name, url in urls.items():
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cadrille-env-check"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read(4096)
            dt = time.time() - t0
            record(f"网络 {name}", "OK", f"{resp.status}, {dt:.1f}s, {len(body)}B")
            any_ok = True
        except Exception as exc:
            record(f"网络 {name}", "WARN", f"{type(exc).__name__}: {exc}")
    if not any_ok:
        record("网络汇总", "FAIL", "两个源都不通, 后面下不了数据/权重")
    else:
        record("网络汇总", "OK",
               "不通的那个源可用 export HF_ENDPOINT=https://hf-mirror.com 替换")


# ----------------------------------------------------------------------------
# 7. Open3D 无头渲染 —— 图像模态的命门
# ----------------------------------------------------------------------------
def render_test():
    """返回 (ok: bool, detail: str)。在子进程里跑, 避免污染主进程 GL 状态。"""
    import numpy as np
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(width=200, height=200, visible=False)
    if not ok:
        return False, "create_window 返回 False"
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    buf = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    arr = np.asarray(buf)
    mean = float(arr.mean())
    if arr.size == 0:
        return False, "捕获到空图"
    if mean <= 1e-6:
        return False, f"渲染全黑(mean={mean:.2e}), 通常是缺 OSMesa/EGL 或驱动问题"
    return True, f"mean={mean:.4f}, shape={arr.shape}"


def check_headless_render(skip=False):
    section("7. Open3D 无头渲染 (图像/多视图模态的命门)")
    if skip:
        record("无头渲染", "SKIP", "--skip-render")
        return
    try:
        import open3d  # noqa
    except Exception as exc:
        record("无头渲染", "SKIP", f"open3d 未安装: {exc}")
        return

    # 依次尝试几种平台设置: 官方依赖 libosmesa6-dev, 所以 osmesa 优先
    candidates = [
        ("osmesa", {"PYOPENGL_PLATFORM": "osmesa"}),
        ("egl", {"PYOPENGL_PLATFORM": "egl"}),
        ("默认(不设置)", {}),
    ]
    ok_any = False
    for name, env in candidates:
        detail = run_render_in_subprocess(env)
        if detail.startswith("OK:"):
            record(f"无头渲染 [{name}]", "OK", detail[3:])
            ok_any = True
            if name != "默认(不设置)":
                record("建议环境变量", "WARN",
                       f"启动前需 export PYOPENGL_PLATFORM={env['PYOPENGL_PLATFORM']}")
            break
        record(f"无头渲染 [{name}]", "WARN", detail)

    if not ok_any:
        record("无头渲染", "FAIL",
               "三种方式都失败。修复: apt-get install -y libosmesa6-dev libgl1-mesa-glx, "
               "并用 -DENABLE_HEADLESS_RENDERING=ON 重新编译 Open3D")


def run_render_in_subprocess(env):
    """把渲染测试放到子进程, 返回 'OK:...' 或错误描述。"""
    import subprocess
    code = (
        "import numpy as np, open3d as o3d\n"
        "m=o3d.geometry.TriangleMesh.create_box(); m.compute_vertex_normals()\n"
        "v=o3d.visualization.Visualizer()\n"
        "assert v.create_window(width=200,height=200,visible=False), 'create_window False'\n"
        "v.add_geometry(m); v.poll_events(); v.update_renderer()\n"
        "a=np.asarray(v.capture_screen_float_buffer(do_render=True)); v.destroy_window()\n"
        "print('OK:mean=%.4f,shape=%s'%(a.mean(),a.shape)) if a.size and a.mean()>1e-6 "
        "else print('FAIL: mean=%s'%(a.mean() if a.size else 'empty'))\n"
    )
    e = dict(os.environ)
    e.update(env)
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=120, env=e)
    except subprocess.TimeoutExpired:
        return "超时 120s"
    out = (p.stdout or "").strip().splitlines()
    out = out[-1] if out else ""
    if out.startswith("OK:"):
        return out
    err = (p.stderr or "").strip().splitlines()
    return f"{out or '无输出'} | {err[-1] if err else ''}"


# ----------------------------------------------------------------------------
# 8. CadQuery 端到端: 执行代码 -> 网格化
#    evaluate.py 里 py_file_to_mesh_and_brep_files 走的就是这条路径
# ----------------------------------------------------------------------------
def check_cadquery():
    section("8. CadQuery 端到端 (生成代码 -> 网格化, 与 evaluate.py 同路径)")
    try:
        import cadquery as cq
        import trimesh
    except Exception as exc:
        record("CadQuery 端到端", "SKIP", f"依赖缺失: {exc}")
        return

    # 用论文 Fig.10 里真实预测过的代码片段(盒体+四个凸台)
    code = (
        "import cadquery as cq\n"
        "w0 = cq.Workplane('XY', origin=(0, 0, 88))\n"
        "r = w0.sketch().push([(-87, -89)]).rect(20, 22).push([(-87, 89)]).rect(20, 22)\n"
        "r = r.push([(88, -89)]).rect(20, 22).push([(88, 90)]).rect(20, 20)\n"
        "r = r.finalize().extrude(-176)\n"
        "r = r.union(w0.workplane(offset=-15 / 2).box(200, 200, 15))\n"
    )
    t0 = time.time()
    try:
        g = {}
        exec(code, g)
        compound = g["r"].val()
        verts, faces = compound.tessellate(0.001, 0.1)
        trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
        dt = time.time() - t0
        record("CadQuery 执行+网格化", "OK", f"{len(verts)} 顶点, {dt:.2f}s")
        if dt > 3.0:
            record("CadQuery 耗时", "WARN",
                   f"单样本 {dt:.2f}s。evaluate.py 的 join(3) 超时会误判为 invalid, IR 会虚高")
        else:
            record("CadQuery 耗时", "OK", f"{dt:.2f}s < evaluate.py 的 3s 超时")
    except Exception as exc:
        record("CadQuery 执行+网格化", "FAIL", f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------
# 9. 真实 mesh 的 CD / IoU 小算例(可选, 需要数据)
# ----------------------------------------------------------------------------
def check_metrics(data_dir, skip=False):
    section("9. CD / IoU 评估链路 (需要 deepcad_test_mesh)")
    if skip:
        record("评估链路", "SKIP", "--skip-metrics")
        return
    mesh_dir = os.path.join(data_dir, "deepcad_test_mesh")
    if not os.path.isdir(mesh_dir):
        record("评估链路", "SKIP", f"{mesh_dir} 不存在(还没下数据, 正常)")
        return
    import glob
    files = sorted(glob.glob(os.path.join(mesh_dir, "*.stl")))
    if not files:
        record("评估链路", "FAIL", f"{mesh_dir} 里没有 .stl")
        return
    try:
        import numpy as np
        import trimesh
        from scipy.spatial import cKDTree

        def cd(a, b, n=8192):
            pa, _ = trimesh.sample.sample_surface(a, n)
            pb, _ = trimesh.sample.sample_surface(b, n)
            da, _ = cKDTree(pa).query(pb, k=1)
            db, _ = cKDTree(pb).query(pa, k=1)
            return float(np.mean(da ** 2) + np.mean(db ** 2))

        m = trimesh.load_mesh(files[0])
        # GT 与自身比: 由于随机采样, CD 不为 0 (论文附录 B: DeepCAD GT 自比 mean CD 0.16)
        self_cd = cd(m, m)
        record("GT 自比 CD", "OK", f"{os.path.basename(files[0])} -> {self_cd * 1000:.3f}e-3 "
                                  f"(论文报 DeepCAD GT 自比约 0.16e-3)")
        record("样本数", "OK", f"{len(files)} 个 .stl" + ("  (应为 8046)" if len(files) != 8046 else ""))
    except Exception as exc:
        record("评估链路", "FAIL", f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------
def summary():
    section("汇总")
    fails = [r for r in RESULT if r[1] == "FAIL"]
    warns = [r for r in RESULT if r[1] == "WARN"]
    oks = [r for r in RESULT if r[1] == "OK"]
    skips = [r for r in RESULT if r[1] == "SKIP"]
    print(f"OK: {len(oks)}   WARN: {len(warns)}   FAIL: {len(fails)}   SKIP: {len(skips)}")

    if fails:
        print("\n必须修掉的问题:")
        for name, _, detail in fails:
            print(f"  [FAIL] {name}: {detail}")
    if warns:
        print("\n建议关注:")
        for name, _, detail in warns:
            print(f"  [WARN] {name}: {detail}")

    print("\n结论:", end=" ")
    if fails:
        print("环境未就绪, 先修上面的 FAIL 项, 不要下数据集。")
    elif warns:
        print("关键项全部通过, WARN 项不阻塞。可以进入第二步: 下 deepcad_test_mesh + 官方权重跑评估。")
    else:
        print("环境完全就绪, 可以进入第二步。")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data", help="数据集根目录 (默认 ./data)")
    ap.add_argument("--skip-render", action="store_true", help="跳过 Open3D 渲染测试")
    ap.add_argument("--skip-metrics", action="store_true", help="跳过 CD/IoU 链路测试")
    args = ap.parse_args()

    print("cadrille 环境自检 - 参考仓库 AIRI-Institute/cadrille, Dockerfile 钉死版本")
    print(f"cwd={os.getcwd()}  data_dir={os.path.abspath(args.data_dir)}")

    check_python()
    check_gpu()
    check_imports()
    check_versions()
    check_resources(args.data_dir)
    check_network()
    check_headless_render(args.skip_render)
    check_cadquery()
    check_metrics(args.data_dir, args.skip_metrics)

    sys.exit(summary())


if __name__ == "__main__":
    main()
