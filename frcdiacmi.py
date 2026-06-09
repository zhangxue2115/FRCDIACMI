import math

import torch
import torch.nn.functional as F
from torch import nn

from frc_attention import FRCAttention


class MultiHeadAttention(nn.Module):
    """DIA 中使用的多头缩放点积注意力。"""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        batch_size = query.size(0)

        value = self.value_projection(value).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key_projection(key).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        query = self.query_projection(query).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attended = self.scaled_dot_product_attention(value, key, query, mask)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        return self.output_projection(attended)

    def scaled_dot_product_attention(self, value, key, query, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            query.size(-1)
        )
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attention_map = F.softmax(scores, dim=-1)
        attention_map = self.dropout(attention_map)
        return torch.matmul(attention_map, value)


class CrossEntityAttention(nn.Module):
    """DIA 中 CMA 和 MCA 共用的跨实体注意力单元。"""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            hidden_dim, num_heads, dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_entity, context_entity, context_mask=None):
        interaction = self.multi_head_attention(
            context_entity,
            context_entity,
            query_entity,
            context_mask,
        )
        return self.norm(query_entity + self.dropout(interaction))


class SelfAttention(nn.Module):
    """DIA 中用于细化实体内部特征结构的 SA 单元。"""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            hidden_dim, num_heads, dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, entity_features, mask=None):
        self_attended = self.multi_head_attention(
            entity_features,
            entity_features,
            entity_features,
            mask,
        )
        return self.norm(entity_features + self.dropout(self_attended))


class DeepInteractiveAttention(nn.Module):
    """
    Deep Interactive Attention (DIA)。

    DIA 由 miRNA/circRNA 自注意力、circRNA-miRNA attention (CMA)
    和 miRNA-circRNA attention (MCA) 组成。
    """

    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.mirna_self_attention = SelfAttention(
            hidden_dim, num_heads, dropout
        )
        self.circrna_self_attention = SelfAttention(
            hidden_dim, num_heads, dropout
        )
        self.cma = CrossEntityAttention(hidden_dim, num_heads, dropout)
        self.mca = CrossEntityAttention(hidden_dim, num_heads, dropout)

    def forward(self, mirna_embedding, circrna_embedding):
        mirna_embedding = mirna_embedding.unsqueeze(1)
        circrna_embedding = circrna_embedding.unsqueeze(1)

        mirna_embedding = self.mirna_self_attention(mirna_embedding)
        circrna_embedding = self.circrna_self_attention(circrna_embedding)

        # MCA：以 miRNA 为 query，聚合 circRNA 信息。
        updated_mirna = self.mca(mirna_embedding, circrna_embedding)
        # CMA：以 circRNA 为 query，聚合 miRNA 信息。
        updated_circrna = self.cma(circrna_embedding, mirna_embedding)

        return updated_mirna.squeeze(1), updated_circrna.squeeze(1)


class FRCDIACMI(nn.Module):
    """
    FRCDIACMI：基于特征重构校准和深度交互注意力的
    circRNA-miRNA 相互作用预测模型。
    """

    def __init__(
        self,
        mirna_feature_dim,
        circrna_feature_dim,
        embedding_dim=128,
        frc_heads=4,
        dia_heads=4,
        dia_dropout=0.1,
        dia_repeat=1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dia_repeat = dia_repeat

        self.mirna_projection = nn.Linear(
            mirna_feature_dim, embedding_dim
        )
        self.circrna_projection = nn.Linear(
            circrna_feature_dim, embedding_dim
        )

        self.frc_attention = FRCAttention(
            embedding_dim=embedding_dim,
            num_heads=frc_heads,
            qkv_bias=True,
        )
        self.deep_interactive_attention = DeepInteractiveAttention(
            embedding_dim,
            dia_heads,
            dia_dropout,
        )

        self.pair_fusion = nn.Linear(2 * embedding_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)

    def fuse_pair_embeddings(
        self, mirna_embedding, circrna_embedding
    ):
        pair_embedding = torch.cat(
            [mirna_embedding, circrna_embedding], dim=1
        )
        return self.pair_fusion(pair_embedding)

    def forward(self, mirna_features, circrna_features):
        mirna_embedding = self.mirna_projection(mirna_features)
        circrna_embedding = self.circrna_projection(circrna_features)

        # FRC-Attention 对联合实体表示执行特征重构与校准。
        entity_sequence = torch.stack(
            [mirna_embedding, circrna_embedding], dim=1
        )
        calibrated_sequence = self.frc_attention(entity_sequence)
        mirna_embedding = calibrated_sequence[:, 0, :]
        circrna_embedding = calibrated_sequence[:, 1, :]

        # DIA 执行 SA、MCA 和 CMA，实现双向深层交互。
        for _ in range(self.dia_repeat):
            mirna_embedding, circrna_embedding = (
                self.deep_interactive_attention(
                    mirna_embedding, circrna_embedding
                )
            )

        pair_embedding = self.fuse_pair_embeddings(
            mirna_embedding, circrna_embedding
        )
        return torch.sigmoid(self.classifier(pair_embedding))
