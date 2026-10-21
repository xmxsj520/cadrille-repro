# cadrille / Table 1 复现 —— 数据集清单

> 本文只列 **Table 1 需要**的数据。每个条目给出：从哪下、放哪、怎么校验。
> 标注说明：✅ = 已从官方 README 核实；⚠️ = 未能联网核实，需你自行确认最新地址。

---

## 0. 先看这个：按阶段决定你要下哪些

| 你的目标 | 必须下的数据集 |
|---|---|
| **阶段 1**：cadrille 那 4 行 + CAD-Recode | ① deepcad_test_mesh ② cad-recode-v1.5 ③ text2cad |
| **阶段 2**：再加 CAD-SIGNet / Text2CAD 两行 baseline | 上面 3 个 + ④ DeepCAD 原始数据 |
| **阶段 3**：其余点云 baseline | 上面 4 个 + ⑤ HNC-CAD 数据集 |
| **阶段 4**：图像模态基线 | 上面 5 个 + 自己渲染 DeepCAD 四视图 |

**不要一次全下。** 阶段 1 的三个下完就能跑出结果。

---

## 1. deepcad_test_mesh —— 评估用 GT（⭐ 无论如何都要）✅

- **来源**：https://huggingface.co/datasets/maksimko123/deepcad_test_mesh
- **内容**：8046 个 `.stl`
- **说明**：由 DeepCAD 官方 `json2pc.py` 生成，**已归一化到单位立方体**（不要自己重新导出，否则 CD/IoU 对不上）
- **放置路径**：`data/deepcad_test_mesh/*.stl`

```bash
cd data
git lfs install
git clone https://hf-mirror.com/datasets/maksimko123/deepcad_test_mesh
ls deepcad_test_mesh/*.stl | wc -l      # 应为 8046
```

---

## 2. cad-recode-v1.5 —— SFT 训练主力 ✅

- **来源**：https://huggingface.co/datasets/filapro/cad-recode-v1.5
- **内容**：约 100 万个 CadQuery `.py` 程序 + 对应 `.stl` + `train.pkl` / `val.pkl`
- **放置路径**：
  ```
  data/cad-recode-v1.5/
  ├── train/batch_00/{0.py, 0.stl, ...}
  ├── train/batch_01/...
  ├── val/{0.py, 0.stl, ...}
  ├── train.pkl
  └── val.pkl
  ```
- **⚠️ 先只拉一个 batch**（全量几十 GB）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -U "huggingface_hub[cli]" hf_transfer

huggingface-cli download filapro/cad-recode-v1.5 --repo-type dataset \
  --include "train/batch_00/*" "train.pkl" "val/*" --local-dir ./data/cad-recode-v1.5
```

- **生成 stl**（如果下载的版本没有 stl，代码仓库里有转换脚本，依赖 Open3D）：

```bash
cd cadrille && python data/cadrecode2mesh.py
```

---

## 3. text2cad —— 文本模态 + DeepCAD 的 CadQuery 代码 ✅

- **来源**：https://huggingface.co/datasets/maksimko123/text2cad
- **内容**：`train.pkl` / `val.pkl` / `test.pkl` + `cadquery/*.py`（几乎覆盖全部 DeepCAD 样例）
- **放置路径**：
  ```
  data/text2cad/
  ├── cadquery/{0.py, ...}
  ├── train.pkl
  ├── val.pkl
  └── test.pkl
  ```
- 上游说明：文本 prompt 来自 Text2CAD 官方仓库，被**略微缩短**过

```bash
git clone https://hf-mirror.com/datasets/maksimko123/text2cad ./data/text2cad
```

---

## 4. DeepCAD 原始数据 —— `D_pit` / `D_p` / `D_i` / `D_t` 各行的训练 ⚠️

- **来源**：https://github.com/ChrisWu1997/DeepCAD + 论文给的 Google Drive 链接
- **为什么需要**：**cadrille 官方没有提供 DeepCAD 训练集**（`data/README.md` 只支持 DeepCAD **test**）。Table 1 里 `cadrille D_pit` 那行、以及所有 `D_p` / `D_i` / `D_t` 的 baseline，训练数据都得你自己凑。
- **要自己做的加工**：
  1. 用官方 `dataset/json2pc.py` 生成点云 → 得到 `D_p`
  2. **自己渲染四视图**（渲染逻辑参考 cadrille 的 `dataset.py` 里 `mesh_to_image`：Open3D、4 个固定 `front` 方向、128px、加 3px 黑边、2×2 拼接）→ 得到 `D_i`
  3. 文本描述用 Text2CAD 的 → 得到 `D_t`
- **⚠️ Text2CAD 的 DeepCAD 训练集**需要按 Text2CAD 仓库说明申请（另有 HuggingFace 备份），不是直接 clone 就有

---

## 5. HNC-CAD 数据集 —— Point-BERT / DINOv2→HNC-CAD 两行 ⚠️

- **用于**：Table 1 里 `Point-BERT→HNC-CAD`、`DINOv2→HNC-CAD` 两行（训练数据标注为 `D_p` / `D_i`，但方法来自 HNC-CAD 工作）
- **来源**：HNC-CAD 官方仓库（自行搜索 "HNC-CAD CAD reconstruction" 确认地址）
- **注意**：Table 1 的这两行没有合并单元格，说明**只用各自模态训练**

---

## 6. Fusion360（Table 1 用不到，Table 2/3 才用）

- **来源**：https://huggingface.co/datasets/maksimko123/fusion360_test_mesh
- 只用于测试，1725 个样例；无 Python CAD 序列标注，所以能用于 RL 但不能用于 SFT

---

## 7. 官方权重

| 权重 | 地址 | 用途 |
|---|---|---|
| SFT（**Table 1 就是它**） | https://huggingface.co/maksimko123/cadrille | 推理评估 |
| RL | https://huggingface.co/maksimko123/cadrille-rl | Table 2/3 |
| 基座 | `Qwen/Qwen2-VL-2B-Instruct` | 训练起始点；`train.py`/`test.py` 会自动下载 |

```bash
huggingface-cli download maksimko123/cadrille --local-dir ./ckpt_sft
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir ./ckpt_qwen
```

---

## 8. Baseline 代码仓库

| 方法 | 仓库 | 备注 |
|---|---|---|
| CAD-Recode | https://github.com/filaPro/cad-recode ⚠️ | 与 cadrille 同源，最好接 |
| Text2CAD | https://github.com/SadilKhan/Text2CAD ✅ | NeurIPS'24 Spotlight，有代码+数据 |
| DeepCAD | https://github.com/ChrisWu1997/DeepCAD ⚠️ | 数据集 + json2pc.py |
| CAD-SIGNet | 搜索 `SadilKhan/CAD-SIGNet` ⚠️ | 有代码+权重 |
| PointNet | 搜索 `charlesq34/pointnet` ⚠️ | 需自己接 CAD 输出头 |
| Point2Cyl | CVPR 2022，搜索仓库 ⚠️ | |
| MultiCAD / TransCAD / PrismCAD / CAD-Diffuser / CADmium / CAD-Coder / Text-to-CadQuery | 各自搜索论文标题 ⚠️ | 需逐个确认是否开源 |
| **CADCrafter** | ❌ **论文明确说代码未公开** | **无法复现**，只能引用 |
| **BERT→DeepCAD** | ❌ 无现成实现 | 要自己搭 |

---

## 9. 下载完的统一校验脚本

在 `cadrille/` 目录下建 `check_data.py`：

```python
import os, glob, pickle

def check(path, desc, cond):
    ok = cond()
    print(("[OK]  " if ok else "[FAIL]") + f" {desc}: {path}")
    return ok

results = []

results.append(check("data/deepcad_test_mesh", "DeepCAD 测试集 stl 数量 = 8046",
    lambda: len(glob.glob("data/deepcad_test_mesh/*.stl")) == 8046))

results.append(check("data/cad-recode-v1.5/train.pkl", "CAD-Recode train.pkl 可读",
    lambda: isinstance(pickle.load(open("data/cad-recode-v1.5/train.pkl", "rb")), (list, dict))))

results.append(check("data/cad-recode-v1.5/train", "CAD-Recode train 至少 1 个 batch",
    lambda: len(glob.glob("data/cad-recode-v1.5/train/batch_*")) >= 1))

def cad_stl_pair():
    py = glob.glob("data/cad-recode-v1.5/train/batch_00/*.py")
    stl = glob.glob("data/cad-recode-v1.5/train/batch_00/*.stl")
    return len(py) > 0 and len(stl) > 0
results.append(check("data/cad-recode-v1.5/train/batch_00", "py 与 stl 成对存在", cad_stl_pair))

results.append(check("data/text2cad/test.pkl", "Text2CAD pkl 三件套齐全",
    lambda: all(os.path.exists(f"data/text2cad/{s}.pkl") for s in ("train", "val", "test"))))

results.append(check("ckpt_sft", "SFT 权重已下载",
    lambda: os.path.exists("ckpt_sft/config.json")))

print(f"\n{sum(results)}/{len(results)} 项通过")
```

---

## 10. 三个最容易出错的地方

1. **目录结构必须严格一致** —— 路径在 `dataset.py` 里写死，错一个字母就 `FileNotFoundError`。
2. **`git-lfs` 必须装**（官方 Dockerfile 里拼错成 `git-lsf`，照抄会失败）。`.stl` 是 LFS 文件，没装 lfs 会下成几十字节的指针文件。
3. **只放数据盘**（AutoDL 是 `/root/autodl-tmp`），系统盘会写满。
