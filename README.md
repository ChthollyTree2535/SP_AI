# SP_AI

基于 PyTorch 的 FAST 单脉冲候选图像分类器，用于区分真实脉冲与噪声 / 射频干扰，并按模型得分排序筛选。输入为 [TransientX](https://github.com/ypmen/TransientX) 生成的 PNG 诊断图，支持 CPU、NVIDIA CUDA 和 Apple Silicon MPS。

## 安装

推荐 Python 3.10 及以上版本。

```bash
git clone https://github.com/ChthollyTree2535/SP_AI.git
cd SP_AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_install.py
```

## 预测

递归评分文件夹中的 PNG，结果按 `prob_real` 降序写入 CSV：

```bash
python predict_transientx.py \
  --model models/best_model.pt \
  --input /path/to/png_folder \
  --output scores.csv \
  --device auto
```

仅将得分严格 **> 0.8** 的图片平铺复制到 `A_plot/`，保留原文件名：

```bash
python copy_scored_flat.py --scores scores.csv --output-dir A_plot --min-prob 0.8
```

原始图片不修改。目标文件重名时停止复制，避免覆盖。默认跳过形如 `0.12345678_` 的已评分文件和名为 `plot` 的子目录，需重新评分时加 `--force`。

## 模型与结果

- **数据集**：18,612 张人工标注图像，正样本 2,267 张，负样本 16,345 张。训练 / 验证 / 测试为 13,028 / 2,792 / 2,792 张，训练图像未包含在本仓库。
- **模型**：DiagnosticResNetV2，残差 CNN + SE 通道注意力，约 113 万参数。默认裁剪区域为 `(68, 184, 647, 950)`，输入高 × 宽为 `256 × 192`，适配原始 `681 × 1021` 诊断图布局。
- **训练**：Focal Loss + AdamW，正样本重采样至 20%，配合裁剪、亮度、对比度及噪声增强，训练 5 轮。

| 测试集指标（阈值 0.5） | 结果 |
| --- | ---: |
| 召回率 | 95.59% |
| 精确率 | 89.78% |
| F1 | 92.59% |
| 平均精度 AP | 98.29% |

以上来自图像级随机划分的测试集，完整记录见 `models/metrics.json`。`prob_real` 是分类得分，未经概率校准；阈值 0.8 的效果应单独验证。

## 重新训练

将负样本放入 `data/0/`，正样本放入 `data/1/`：

```bash
python train_transientx.py --data-dir data --output-dir runs \
  --run-name my_model --device auto --epochs 5 --batch-size 48 \
  --train-pos-fraction 0.2 --focal-alpha 0.7
```

训练会保存权重、数据划分和评价结果。`evaluate_run.py` 可复评训练记录，`summarize_runs.py` 可汇总多次训练。代码和预训练权重采用 [MIT License](LICENSE)。
