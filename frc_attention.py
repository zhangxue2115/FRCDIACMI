import math

import torch
from torch import nn


class MultiHeadFeatureReconstruction(nn.Module):
    """通过多头自注意力重构联合实体特征。"""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv_projection = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, entity_sequence):
        batch_size, num_entities, embedding_dim = entity_sequence.shape
        qkv = self.qkv_projection(entity_sequence).reshape(
            batch_size,
            num_entities,
            3,
            self.num_heads,
            embedding_dim // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention_map = (
            query @ key.transpose(-2, -1)
        ) * self.scale
        attention_map = attention_map.softmax(dim=-1)
        attention_map = self.attention_dropout(attention_map)

        reconstructed = (
            attention_map @ value
        ).transpose(1, 2).reshape(
            batch_size, num_entities, embedding_dim
        )
        reconstructed = self.output_projection(reconstructed)
        return self.projection_dropout(reconstructed)


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, features):
        return self.relu(features + 3) / 6


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, features):
        return features * self.sigmoid(features)


class ChannelCalibration(nn.Module):
    """根据全局通道统计量重新校准特征响应。"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Linear(channels, channels // reduction)
        self.activation = nn.ReLU(inplace=True)
        self.expand = nn.Linear(channels // reduction, channels)
        self.gate = HardSigmoid()

    def forward(self, features):
        batch_size, channels, _, _ = features.size()
        channel_weights = self.global_pool(features).view(
            batch_size, channels
        )
        channel_weights = self.reduce(channel_weights)
        channel_weights = self.activation(channel_weights)
        channel_weights = self.expand(channel_weights)
        channel_weights = self.gate(channel_weights)
        channel_weights = channel_weights.view(
            batch_size, channels, 1, 1
        )
        return features * channel_weights


class FeatureCalibration(nn.Module):
    """
    FRC-Attention 的特征校准阶段。

    通过逐点卷积、深度可分离卷积和通道重标定抑制噪声，
    并使用残差连接保持原始语义信息。
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        stride=1,
        expansion_ratio=4.0,
        reduction=4,
    ):
        super().__init__()
        hidden_dim = int(input_dim * expansion_ratio)

        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            HardSwish(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            HardSwish(),
            ChannelCalibration(hidden_dim, reduction=reduction),
            nn.Conv2d(hidden_dim, output_dim, 1, bias=False),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, features):
        return features + self.layers(features)


class FRCAttentionBlock(nn.Module):
    """
    Feature Reconstruction Calibration Attention Block。

    先通过多头自注意力重构联合特征，再通过深度可分离卷积
    和通道校准稳定特征表示。
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        dropout=0.1,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.reconstruction_norm = nn.LayerNorm(
            embedding_dim, eps=1e-6
        )
        self.feature_reconstruction = MultiHeadFeatureReconstruction(
            embedding_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.feature_calibration = FeatureCalibration(
            embedding_dim,
            embedding_dim,
        )

    def forward(self, entity_sequence):
        residual = entity_sequence
        reconstructed = self.feature_reconstruction(
            self.reconstruction_norm(entity_sequence)
        )
        reconstructed = reconstructed + residual

        batch_size, num_entities, embedding_dim = (
            reconstructed.shape
        )
        calibration_size = int(math.sqrt(num_entities - 1))

        reference_embedding, calibrated_embedding = torch.split(
            reconstructed,
            [1, num_entities - 1],
            dim=1,
        )
        calibrated_embedding = calibrated_embedding.transpose(
            1, 2
        ).reshape(
            batch_size,
            embedding_dim,
            calibration_size,
            calibration_size,
        )
        calibrated_embedding = self.feature_calibration(
            calibrated_embedding
        ).flatten(2).transpose(1, 2)

        return torch.cat(
            [reference_embedding, calibrated_embedding], dim=1
        )


class FRCAttention(nn.Module):
    """Feature Reconstruction Calibration Attention (FRC-Attention)."""

    def __init__(
        self,
        embedding_dim=768,
        num_heads=4,
        qkv_bias=True,
        qk_scale=None,
        dropout=0.1,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.frc_block = FRCAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.output_norm = nn.LayerNorm(embedding_dim, eps=1e-5)

    def forward(self, entity_sequence):
        calibrated_sequence = self.frc_block(entity_sequence)
        return self.output_norm(calibrated_sequence)
