import math
from typing import Optional

import torch
from torch import nn


class ContextualNormalizer(nn.Module):
    """
    Normalizes neural features across days/blocks using a lightweight transformer encoder.
    """

    def __init__(
        self,
        neural_dim: int,
        n_days: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        ff_multiplier: int = 2,
        dropout: float = 0.1,
        max_blocks: int = 512,
        use_block_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_block_embedding = use_block_embedding and max_blocks > 0

        self.neural_proj = nn.Linear(neural_dim, hidden_dim)
        self.day_embedding = nn.Embedding(n_days, hidden_dim)
        if self.use_block_embedding:
            self.block_embedding = nn.Embedding(max_blocks, hidden_dim)
        else:
            self.block_embedding = None

        meta_input_dim = hidden_dim * (1 + int(self.use_block_embedding))
        self.meta_proj = nn.Linear(meta_input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_multiplier * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(hidden_dim, neural_dim)
        self.residual_norm = nn.LayerNorm(neural_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
        block_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, time, neural_dim]
            day_idx: Tensor of shape [batch]
            block_idx: Optional tensor of shape [batch]
        """
        orig_dtype = x.dtype
        batch_size, seq_len, _ = x.shape

        # Detect padding (assumes zero padding).
        padding_mask = (x.abs().sum(dim=-1) == 0)
        if not padding_mask.any():
            padding_mask = None

        # Project neural features.
        h = self.neural_proj(x)

        # Day embeddings.
        day_emb = self.day_embedding(day_idx.long()).unsqueeze(1).expand(-1, seq_len, -1)
        meta_features = [day_emb]

        # Block embeddings if available.
        if self.use_block_embedding and block_idx is not None:
            clamped_blocks = torch.clamp(
                block_idx.long(),
                min=0,
                max=self.block_embedding.num_embeddings - 1,
            )
            block_emb = self.block_embedding(clamped_blocks).unsqueeze(1).expand(-1, seq_len, -1)
            meta_features.append(block_emb)

        meta = torch.cat(meta_features, dim=-1)
        meta = self.meta_proj(meta)
        h = h + meta

        # Positional encoding (sinusoidal) computed on the fly.
        pos_encoding = self._build_sinusoidal_encoding(seq_len, self.hidden_dim, x.device, h.dtype)
        h = h + pos_encoding

        # Transformer expects float32 for stability; convert back afterwards.
        h = h.to(torch.float32)
        if padding_mask is not None:
            encoder_out = self.encoder(h, src_key_padding_mask=padding_mask)
        else:
            encoder_out = self.encoder(h)
        encoder_out = encoder_out.to(orig_dtype)

        # Project back to neural_dim and add residual.
        delta = self.output_proj(encoder_out)
        delta = self.dropout(delta)
        return self.residual_norm(x + delta)

    @staticmethod
    def _build_sinusoidal_encoding(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))

        pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


