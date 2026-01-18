import torch
import torch.nn as nn
from torchvision import models

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        cnn_out_dim: int = 512,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        backbone = models.resnet18(weights=None)  # must match training
        self.cnn_stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.cnn_layers = nn.Sequential(
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_proj = nn.Linear(cnn_out_dim, embed_dim)
        self.token_proj = nn.Linear(cnn_out_dim, embed_dim)

        feat_h = img_size // 32
        feat_w = img_size // 32
        num_tokens = feat_h * feat_w

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        fusion_dim = 2 * embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.cnn_stem(x)
        feat = self.cnn_layers(x)  # [B, 512, 7, 7]
        B, C, H, W = feat.shape

        cnn_global = self.cnn_pool(feat).view(B, C)
        cnn_global = self.cnn_proj(cnn_global)  # [B, D]

        tokens = feat.flatten(2).transpose(1, 2)  # [B, N, C]
        tokens = self.token_proj(tokens)          # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)    # [B, 1, D]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, 1+N, D]

        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.pos_drop(tokens)

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)
        trans_cls = tokens[:, 0]                  # [B, D]

        fused = torch.cat([cnn_global, trans_cls], dim=1)
        logits = self.classifier(fused)
        return logits
