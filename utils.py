import torch
import numpy as np
from sklearn.model_selection import train_test_split


# def generate_data(adj, mirna_features, circrna_features):
#     pos_data = [[mirna_features[i], circrna_features[j]] for i in range(len(adj)) for j in range(len(adj[0])) if adj[i][j] == 1]

#     neg_data = []
#     while len(neg_data) < len(pos_data):
#         x = np.random.randint(len(mirna_features))
#         y = np.random.randint(len(circrna_features))
#         if adj[x][y] == 0:
#             neg_data.append([mirna_features[x], circrna_features[y]])

#     return pos_data, neg_data


def generate_data(adj, mirna_features, circrna_features, neg_ratio=1.0):
    """
    adj: miRNA-circRNA 邻接矩阵 (num_miRNAs, num_circRNAs)
    mirna_features: miRNA 特征
    circrna_features: circRNA 特征
    neg_ratio: 负样本 / 正样本 比例
    """

    # 1️⃣ 生成正样本
    pos_data = [
        [mirna_features[i], circrna_features[j]]
        for i in range(len(adj))
        for j in range(len(adj[0]))
        if adj[i][j] == 1
    ]

    num_pos = len(pos_data)
    num_neg = int(num_pos * neg_ratio)

    # 2️⃣ 生成负样本
    neg_data = []
    used_pairs = set()  # 防止重复采样

    while len(neg_data) < num_neg:
        x = np.random.randint(len(mirna_features))
        y = np.random.randint(len(circrna_features))

        if adj[x][y] == 0 and (x, y) not in used_pairs:
            neg_data.append([mirna_features[x], circrna_features[y]])
            used_pairs.add((x, y))

    return pos_data, neg_data

def split_data(pos_data, neg_data, train_ratio=0.8, val_ratio=0., test_ratio=0.2):
    data = pos_data + neg_data
    labels = [1] * len(pos_data) + [0] * len(neg_data)

    train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=(1 - train_ratio), random_state=42)
    val_data, test_data, val_labels, test_labels = temp_data, temp_data, temp_labels, temp_labels

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def data_to_tensor(data):
    mirna = []
    circrna = []
    for mirna_features, circrna_features in data:
        mirna.append(mirna_features)
        circrna.append(circrna_features)
    return torch.Tensor(mirna).cuda(), torch.Tensor(circrna).cuda()
