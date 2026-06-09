# FRCDIACMI 核心模型代码

论文名称：

> FRCDIACMI: Prediction of circRNA-miRNA Interaction based on Feature Reconstruction Calibration and Deep Interactive Attention


## 文件结构

```text
main.py           五折交叉验证训练与评价入口
frcdiacmi.py      FRCDIACMI 主模型及 DIA 模块
frc_attention.py FRC-Attention 模块
utils.py          样本生成和张量转换工具
README_CN.md      本说明文件
```

## 模型结构

```text
RNA-BERTa feature representations
                |
                v
Independent linear projections
                |
                v
FRC-Attention
├── Multi-Head Attention-based Feature Reconstruction
└── Feature Calibration
    ├── Point-wise convolution
    ├── Depth-wise separable convolution
    └── Channel calibration
                |
                v
Deep Interactive Attention (DIA)
├── Self-Attention (SA)
├── circRNA-miRNA Attention (CMA)
└── miRNA-circRNA Attention (MCA)
                |
                v
Pair feature concatenation and fusion
                |
                v
Binary interaction prediction
```

## 代码命名对应关系

| 论文名称 | 代码名称 |
|---|---|
| FRCDIACMI | `FRCDIACMI` |
| Feature Reconstruction Calibration Attention | `FRCAttention` |
| Multi-Head Attention-based Feature Reconstruction | `MultiHeadFeatureReconstruction` |
| Feature Calibration | `FeatureCalibration` |
| Deep Interactive Attention | `DeepInteractiveAttention` |
| Self-Attention | `SelfAttention` |
| circRNA-miRNA Attention | `cma` |
| miRNA-circRNA Attention | `mca` |


## 数据输入

`main.py` 读取以下三个文件：

```text
interaction_matrix.csv
miRNA_feature.csv
circRNA_feature.csv
```


## 运行环境

主要依赖：

```text
Python 3.9
PyTorch 1.12.0
NumPy 1.23
Pandas 1.3.5
scikit-learn 1.2.2
```

CUDA 和 PyTorch 版本应根据目标服务器显卡驱动选择兼容组合。

## 运行方法

```bash
python main.py
```
