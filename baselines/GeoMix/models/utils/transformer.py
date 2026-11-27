import torch
import torch.nn as nn

class BaseTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer=4, num_heads=8, activation=nn.SiLU(), residual=False, last_act=False, pooling=True, pe=False, max_seq_len=512):
        super(BaseTransformer, self).__init__()
        
        self.residual = residual
        self.num_layer = num_layer
        self.pooling = pooling
        self.pe = pe

        if self.pe:
            self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, 
                                               dim_feedforward=hidden_dim*4, batch_first=True)
        self.xfmr = nn.TransformerEncoder(enc_layer, num_layer)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.activation = activation
        self.last_act = last_act
        if last_act:
            self.last_activation = activation
        else:
            self.last_activation = nn.Identity()
    
    def forward(self, x):
        if self.pe:
            seq_len = x.size(1)
            x = x + self.positional_encoding[:, :seq_len, :]

        x = self.xfmr(x)

        if self.pooling:
            x = x.mean(dim=1)

        x = self.fc_out(x)
        x = self.last_activation(x)

        return x

