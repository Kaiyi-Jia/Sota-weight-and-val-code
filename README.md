# Sota-weight-and-val-code
Sota-weight and val code of Boosting Representation Learning for High-Level Semantic Information in Facial Expression Recognition
## 📦 权重文件

预训练权重通过 [GitHub Releases](../../releases) 提供，请从 **Latest Release** 下载所需文件。

| 文件名 | 任务 | 训练数据集 |
|--------|------|------------|
| `sota_ferplus.pth` | 7 分类 | FERPlus |
| `sota_raf-db.pth` | 7 分类 | RAF‑DB |
| `sota_affect7.pth` | 7 分类 | AffectNet (7 类) |
| `sota_affect8.pth` | 8 分类 | AffectNet (8 类) |

---

## 🧠 情绪类别

### 7 分类
0: Neutral 1: Happy 2: Sad 3: Surprised
4: Scared 5: Disgusted 6: Angry
### 8 分类（在 7 类基础上增加）
7: Contemptuous

## ⚙️ 环境要求

- Python 3.8+
- PyTorch ≥ 1.10（推荐 CUDA 11.7+）
- OpenAI CLIP（官方库）
- torchvision, tqdm, numpy

🚀 快速开始
1. 下载权重
从 Releases 页面下载所需的 .pth 文件，放入项目根目录。

2. 准备测试数据
测试集根目录下需包含按类别编号命名的子文件夹，例如：

text
D:/val/
├── 0/
├── 1/
├── ...
└── 6/   (7 分类) 或 7/ (8 分类)
3. 运行评估
测试单个权重（7 分类）
bash
python evaluate.py --weights sota_affect7.pth --test_root D:/val --num_classes 7
测试多个七分类权重
bash
python evaluate.py \
    --weights sota_ferplus.pth sota_raf-db.pth sota_affect7.pth \
    --test_root D:/val \
    --num_classes 7
测试八分类权重
bash
python evaluate.py --weights sota_affect8.pth --test_root D:/val --num_classes 8
导出结果到 CSV
bash
python evaluate.py --weights sota_affect7.pth sota_affect8.pth \
    --test_root D:/val --num_classes 7 --output results.csv
命令行参数
参数	类型	说明	默认值
--weights	多路径	必须，一个或多个权重文件路径	-
--test_root	路径	必须，测试集根目录	-
--num_classes	7 或 8	分类数	8
--batch_size	int	批次大小	64
--num_workers	int	数据加载线程数	0
--device	cuda/cpu	运行设备	cuda (若可用)
--output	路径	可选的 CSV 结果保存路径	None
📊 输出示例
text
Testing weight: sota_affect7.pth
✅ Weights loaded successfully.
Valid test samples: 4000 (classes 0~6)
🔥 Starting evaluation...
🎯 Overall Accuracy (7-class): 67.85% (2714/4000)

==================================================
Per-Class Accuracy:
  [0] Neutral     : 72.50% (580/800)
  [1] Happy       : 85.33% (683/800)
  [2] Sad         : 61.25% (490/800)
  [3] Surprised   : 76.13% (609/800)
  [4] Scared      : 55.00% (440/800)
  [5] Disgusted   : 52.88% (423/800)
  [6] Angry       : 62.63% (501/800)
==================================================
📁 项目结构
text
.
├── evaluate.py                # 主评估脚本
├── README.md
└── (权重文件需从 Release 下载)
