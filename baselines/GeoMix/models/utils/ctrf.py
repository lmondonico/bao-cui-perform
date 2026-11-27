import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attn_temperature=1.0):
        super().__init__()
        self.attn_temperature = attn_temperature
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        q = k = v = x
        attn_output, _ = self.self_attn(q, k, v, need_weights=False)
        attn_output = attn_output / self.attn_temperature

        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)

        return x


class BaseTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layer=4,
        num_heads=8,
        activation=nn.SiLU(),
        residual=False,
        last_act=False,
        pooling=True,
        pe=False,
        max_seq_len=512,
        attn_dropout=0.1,
        attn_temperature=1.0,
    ):
        super(BaseTransformer, self).__init__()

        self.residual = residual
        self.num_layer = num_layer
        self.pooling = pooling
        self.pe = pe

        if self.pe:
            self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        self.xfmr = nn.Sequential(*[
            CustomTransformerLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=attn_dropout,
                attn_temperature=attn_temperature
            )
            for _ in range(num_layer)
        ])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.activation = activation
        self.last_act = last_act
        self.last_activation = activation if last_act else nn.Identity()

    def forward(self, x):
        x = self.input_proj(x)

        if self.pe:
            seq_len = x.size(1)
            x = x + self.positional_encoding[:, :seq_len, :]

        x = self.xfmr(x)

        if self.pooling:
            x = x.mean(dim=1)

        x = self.fc_out(x)
        x = self.last_activation(x)

        return x
