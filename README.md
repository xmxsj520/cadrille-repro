# cadrille-repro

复现论文 **[cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning](https://arxiv.org/abs/2505.22914)** 中 Table 1 的 `cadrille R_pi` 行。

## 复现结果

使用官方 SFT 权重，在 DeepCAD 测试集（8046 个样本）上评估，三个模态全部完成。

| 模态 | median CD↓ | mean IoU↑ | IR↓ | 论文 CD / IoU / IR |
|---|---|---|---|---|
| **点云** | **0.180** | **87.3** | **1.86** | 0.18 / 87.1 / 2.1 |
| **图像** | **0.184** | **86.6** | **1.44** | 0.18 / 86.1 / 1.5 |
| **文本** | **0.202** | **82.3** | **2.56** | 0.20 / 82.1 / 1.4 |

**9 个指标中 8 个与论文一致**（CD 差异 ≤ 0.004，IoU 差异 ≤ 0.5）。

> 完整报告见 [复现报告.md](复现报告.md)

## 关键发现

官方 `evaluate.py` 将每个生成脚本的执行超时硬编码为 **3 秒**（`process.join(3)`）。在较慢的 CPU 上，可正常执行的样本会被超时终止并计为 invalid，导致 **IR 系统性虚高约 0.5 个百分点**。

| 超时设置 | 点云 IR | 图像 IR | 文本 IR |
|---|---|---|---|
| 3 秒（官方默认） | 2.36 | 2.01 | — |
| **10 秒** | **1.86** | **1.44** | 2.56 |
| 30 秒 | — | — | 2.54 |
| 论文 | 2.1 | 1.5 | 1.4 |

点云与图像的 IR 偏差由超时造成；文本的 IR 偏差（2.56 vs 1.4）无法由超时解释，推测与 CadQuery/OCP 版本差异或 `sdpa` 替代 `flash-attn` 有关。

## 复现范围

| 项目 | 范围 |
|---|---|
| 复现的行 | 1 行 / 20 行（`cadrille R_pi`） |
| 复现的模态 | 点云、多视图图像、文本（全部） |
| 复现的指标 | 9 个 |
| 未做 | 16 个 baseline、`D_pit` 行、RL 微调、从零训练 |

## 环境

| 项 | 配置 |
|---|---|
| GPU | NVIDIA RTX 4090 D，24 GB（sm_89） |
| CPU / 内存 | 16 vCPU / 62 GB |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu124 / CUDA 12.4 |
| Open3D | 0.18.0+8e43455（**源码编译，headless 渲染**） |
| pytorch3d | 0.7.8（源码编译，`TORCH_CUDA_ARCH_LIST=8.9`） |
| 注意力实现 | `sdpa`（未安装 flash-attn） |

## 仓库内容

| 文件 | 说明 |
|---|---|
| [复现报告.md](复现报告.md) | 完整复现报告：环境、流程、结果、12 条工程踩坑记录、命令清单 |
| [datasets.md](datasets.md) | 数据集清单：来源、路径要求、校验方法 |
| [setup_env.sh](setup_env.sh) | 环境一键安装脚本（8 步、可重入） |
| [step1_check_env.py](step1_check_env.py) | 环境自检（9 项：GPU/bf16/渲染/CadQuery/指标链路） |
| [eval_adapter.py](eval_adapter.py) | 统一评估适配器（把任意 baseline 预测转成 cadrille 评估口径） |

## 快速开始

```bash
git clone https://github.com/AIRI-Institute/cadrille.git
cd cadrille

# 上传本仓库的 setup_env.sh / step1_check_env.py 到 cadrille/ 目录
bash setup_env.sh --skip-flash          # 装环境，1–2.5 小时
python step1_check_env.py --data-dir ./data   # 自检

sed -i "s/flash_attention_2/sdpa/g" train.py test.py   # 未装 flash-attn 时
```

数据下载与环境细节见 [复现报告.md](复现报告.md) 附录 A。

## 引用

```bibtex
@article{kolodiazhnyi2025cadrille,
  title={cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning},
  author={Kolodiazhnyi, Maksim and Tarasov, Denis and Zhemchuzhnikov, Dmitrii and
          Nikulin, Alexander and Zisman, Ilya and Vorontsova, Anna and
          Konushin, Anton and Kurenkov, Vladislav and Rukhovich, Danila},
  journal={arXiv preprint arXiv:2505.22914},
  year={2025}
}
```
