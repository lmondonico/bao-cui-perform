import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MOTIFAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = q.size(0)

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, adjoin_matrix, dist_matrix
        )

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)

        return output, attention_weights


class MOTIFLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate):
        super().__init__()
        self.multi_head_att_1 = MOTIFAttention(d_model // 2, num_heads)
        self.multi_head_att_2 = MOTIFAttention(d_model // 2, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = torch.chunk(x, 2, dim=-1)

        x_l, attention_weights_local = self.multi_head_att_1(
            x1, x1, x1, encoder_padding_mask, adjoin_matrix, dist_matrix=None
        )

        x_g, attention_weights_global = self.multi_head_att_2(
            x2,
            x2,
            x2,
            encoder_padding_mask,
            adjoin_matrix=None,
            dist_matrix=dist_matrix,
        )

        attn_output = torch.cat([x_l, x_g], dim=-1)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2, attention_weights_local, attention_weights_global


class MOTIF(nn.Module):
    def __init__(self, input_vocab_size, d_model, dff, num_layers, num_heads, rate):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.dropout = nn.Dropout(rate)
        self.encoder_layers = nn.ModuleList(
            [MOTIFLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )
        self.moiety_linear = nn.Linear(d_model, d_model)

    def forward(self, x, adjoin_matrix=None, dist_matrix=None):
        motif_padding_mask = create_padding_mask(x)

        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)

        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.dropout(x)
        x = self.moiety_linear(x)

        motif_attention_weights_list_local = []
        motif_attention_weights_list_global = []

        for i in range(self.num_layers):
            x, motif_attention_weights_local, motif_attention_weights_global = (
                self.encoder_layers[i](
                    x,
                    motif_padding_mask,
                    adjoin_matrix,
                    dist_matrix=dist_matrix,
                )
            )
            motif_attention_weights_list_local.append(motif_attention_weights_local)
            motif_attention_weights_list_global.append(motif_attention_weights_global)

        return (
            x,
            motif_attention_weights_list_local,
            motif_attention_weights_list_global,
            motif_padding_mask,
        )


class MolarRatioEncoder(nn.Module):
    def __init__(self, d_model, hidden_dim=64):
        super(MolarRatioEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, molar_ratios):
        mr_reshaped = molar_ratios.unsqueeze(-1)
        mr_weights = self.encoder(mr_reshaped)
        zero_mask = (molar_ratios == 0).unsqueeze(-1)
        mr_weights = mr_weights.masked_fill(zero_mask, 0.0)
        return mr_weights


class TemperatureEncoder(nn.Module):
    def __init__(self, d_model, hidden_dim=32):
        super(TemperatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, temperature):
        temp_reshaped = temperature.unsqueeze(-1)
        temp_weights = self.encoder(temp_reshaped)
        return temp_weights


class FormulationAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(FormulationAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_mask = ~mask
        attended_x, attention_weights = self.attention(
            query=x, key=x, value=x, key_padding_mask=attn_mask
        )
        x = self.layer_norm(x + self.dropout(attended_x))
        return x, attention_weights


class GlobalPooling(nn.Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()

    def forward(self, x, mask):
        mask_expanded = mask.unsqueeze(-1)
        masked_x = x * mask_expanded
        pooled = masked_x.sum(dim=1)
        valid_count = mask.sum(dim=1, keepdim=True).float()
        pooled = pooled / (valid_count + 1e-8)
        return pooled


def create_padding_mask(batch_data):
    padding_mask = (batch_data == 0).float()
    return padding_mask.unsqueeze(1).unsqueeze(2)


def rescale_distance_matrix(w):
    constant_value = torch.tensor(1.0, dtype=torch.float32)
    return (constant_value + torch.exp(constant_value)) / (
        constant_value + torch.exp(constant_value - w)
    )


def feed_forward_network(d_model, dff):
    return nn.Sequential(nn.Linear(d_model, dff), nn.GELU(), nn.Linear(dff, d_model))


def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix):
    """
    Compute scaled dot-product attention with structural biases.

    Local attention (adjoin_matrix): Uses adjacency mask for graph structure
    Global attention (dist_matrix): Uses distance weighting with ReLU
    """
    dk = q.size()[-1]

    if dist_matrix is not None:
        matmul_qk = F.relu(torch.matmul(q, k.transpose(-2, -1)))
        dist_matrix_rescaled = rescale_distance_matrix(dist_matrix)
        scaled_attention_logits = (matmul_qk * dist_matrix_rescaled) / math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += mask * -1e9

    else:
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        if adjoin_matrix is not None:
            scaled_attention_logits += adjoin_matrix

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


def scaled_dot_product_attention_no_relu(q, k, v, mask, adjoin_matrix, dist_matrix):
    """Same as scaled_dot_product_attention but WITHOUT ReLU in the global branch."""
    dk = q.size()[-1]

    if dist_matrix is not None:
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dist_matrix_rescaled = rescale_distance_matrix(dist_matrix)
        scaled_attention_logits = (matmul_qk * dist_matrix_rescaled) / math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += mask * -1e9

    else:
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        if adjoin_matrix is not None:
            scaled_attention_logits += adjoin_matrix

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MOTIFAttentionNoReLU(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MOTIFAttentionNoReLU, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = q.size(0)

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention_no_relu(
            q, k, v, mask, adjoin_matrix, dist_matrix
        )

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)

        return output, attention_weights


class MOTIFLayerNoReLU(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate):
        super(MOTIFLayerNoReLU, self).__init__()
        self.multi_head_att_1 = MOTIFAttention(d_model // 2, num_heads)
        self.multi_head_att_2 = MOTIFAttentionNoReLU(d_model // 2, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = torch.chunk(x, 2, dim=-1)

        x_l, attention_weights_local = self.multi_head_att_1(
            x1, x1, x1, encoder_padding_mask, adjoin_matrix, dist_matrix=None
        )

        x_g, attention_weights_global = self.multi_head_att_2(
            x2,
            x2,
            x2,
            encoder_padding_mask,
            adjoin_matrix=None,
            dist_matrix=dist_matrix,
        )

        attn_output = torch.cat([x_l, x_g], dim=-1)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2, attention_weights_local, attention_weights_global


class MOTIFNoReLU(nn.Module):
    def __init__(self, input_vocab_size, d_model, dff, num_layers, num_heads, rate):
        super(MOTIFNoReLU, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.dropout = nn.Dropout(rate)
        self.encoder_layers = nn.ModuleList(
            [MOTIFLayerNoReLU(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )
        self.moiety_linear = nn.Linear(d_model, d_model)

    def forward(self, x, adjoin_matrix=None, dist_matrix=None):
        motif_padding_mask = create_padding_mask(x)

        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)

        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.dropout(x)
        x = self.moiety_linear(x)

        motif_attention_weights_list_local = []
        motif_attention_weights_list_global = []

        for i in range(self.num_layers):
            x, motif_attention_weights_local, motif_attention_weights_global = (
                self.encoder_layers[i](
                    x,
                    motif_padding_mask,
                    adjoin_matrix,
                    dist_matrix=dist_matrix,
                )
            )
            motif_attention_weights_list_local.append(motif_attention_weights_local)
            motif_attention_weights_list_global.append(motif_attention_weights_global)

        return (
            x,
            motif_attention_weights_list_local,
            motif_attention_weights_list_global,
            motif_padding_mask,
        )


def scaled_dot_product_attention_hadamard(q, k, v, mask, adjoin_matrix, dist_matrix):
    """Same as scaled_dot_product_attention but uses Hadamard (element-wise)
    masking in the local branch instead of hard additive masking."""
    dk = q.size()[-1]

    if dist_matrix is not None:
        matmul_qk = F.relu(torch.matmul(q, k.transpose(-2, -1)))
        dist_matrix_rescaled = rescale_distance_matrix(dist_matrix)
        scaled_attention_logits = (matmul_qk * dist_matrix_rescaled) / math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += mask * -1e9

    else:
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))

        if adjoin_matrix is not None:
            raw_adj = (adjoin_matrix > -1e8).float()
            scaled_attention_logits = (matmul_qk * raw_adj) / math.sqrt(dk)
        else:
            scaled_attention_logits = matmul_qk / math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += mask * -1e9

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MOTIFAttentionHadamard(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MOTIFAttentionHadamard, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = q.size(0)

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention_hadamard(
            q, k, v, mask, adjoin_matrix, dist_matrix
        )

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)

        return output, attention_weights


class MOTIFLayerHadamard(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate):
        super(MOTIFLayerHadamard, self).__init__()
        self.multi_head_att_1 = MOTIFAttentionHadamard(d_model // 2, num_heads)
        self.multi_head_att_2 = MOTIFAttention(d_model // 2, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = torch.chunk(x, 2, dim=-1)

        x_l, attention_weights_local = self.multi_head_att_1(
            x1, x1, x1, encoder_padding_mask, adjoin_matrix, dist_matrix=None
        )

        x_g, attention_weights_global = self.multi_head_att_2(
            x2,
            x2,
            x2,
            encoder_padding_mask,
            adjoin_matrix=None,
            dist_matrix=dist_matrix,
        )

        attn_output = torch.cat([x_l, x_g], dim=-1)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2, attention_weights_local, attention_weights_global


class MOTIFHadamard(nn.Module):
    def __init__(self, input_vocab_size, d_model, dff, num_layers, num_heads, rate):
        super(MOTIFHadamard, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.dropout = nn.Dropout(rate)
        self.encoder_layers = nn.ModuleList(
            [MOTIFLayerHadamard(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )
        self.moiety_linear = nn.Linear(d_model, d_model)

    def forward(self, x, adjoin_matrix=None, dist_matrix=None):
        motif_padding_mask = create_padding_mask(x)

        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)

        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.dropout(x)
        x = self.moiety_linear(x)

        motif_attention_weights_list_local = []
        motif_attention_weights_list_global = []

        for i in range(self.num_layers):
            x, motif_attention_weights_local, motif_attention_weights_global = (
                self.encoder_layers[i](
                    x,
                    motif_padding_mask,
                    adjoin_matrix,
                    dist_matrix=dist_matrix,
                )
            )
            motif_attention_weights_list_local.append(motif_attention_weights_local)
            motif_attention_weights_list_global.append(motif_attention_weights_global)

        return (
            x,
            motif_attention_weights_list_local,
            motif_attention_weights_list_global,
            motif_padding_mask,
        )


def scaled_dot_product_attention_vanilla(q, k, v, mask, adjoin_matrix, dist_matrix):
    """Plain scaled dot-product attention. Ignores adjacency and distance matrices."""
    dk = q.size()[-1]

    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    scaled_attention_logits = matmul_qk / math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MOTIFAttentionVanilla(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MOTIFAttentionVanilla, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = q.size(0)

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention_vanilla(
            q, k, v, mask, adjoin_matrix, dist_matrix
        )

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)

        return output, attention_weights


class MOTIFLayerVanilla(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate):
        super(MOTIFLayerVanilla, self).__init__()
        self.proj_down = nn.Linear(d_model, d_model // 2)
        self.multi_head_att = MOTIFAttentionVanilla(d_model // 2, num_heads)
        self.proj_up = nn.Linear(d_model // 2, d_model)
        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x_down = self.proj_down(x)
        attn_output, attention_weights = self.multi_head_att(
            x_down, x_down, x_down, encoder_padding_mask, adjoin_matrix=None, dist_matrix=None
        )
        attn_output = self.proj_up(attn_output)

        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2, attention_weights, attention_weights


class MOTIFVanilla(nn.Module):
    def __init__(self, input_vocab_size, d_model, dff, num_layers, num_heads, rate):
        super(MOTIFVanilla, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.dropout = nn.Dropout(rate)
        self.encoder_layers = nn.ModuleList(
            [MOTIFLayerVanilla(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )
        self.moiety_linear = nn.Linear(d_model, d_model)

    def forward(self, x, adjoin_matrix=None, dist_matrix=None):
        motif_padding_mask = create_padding_mask(x)

        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.dropout(x)
        x = self.moiety_linear(x)

        motif_attention_weights_list_local = []
        motif_attention_weights_list_global = []

        for i in range(self.num_layers):
            x, attn_weights_a, attn_weights_b = self.encoder_layers[i](
                x, motif_padding_mask, adjoin_matrix, dist_matrix=dist_matrix,
            )
            motif_attention_weights_list_local.append(attn_weights_a)
            motif_attention_weights_list_global.append(attn_weights_b)

        return (
            x,
            motif_attention_weights_list_local,
            motif_attention_weights_list_global,
            motif_padding_mask,
        )


def scaled_dot_product_attention_additive_dist(
    q, k, v, mask, adjoin_matrix, dist_matrix
):
    """Same as scaled_dot_product_attention except the global branch uses
    an additive D' bias inside the softmax, without ReLU on QK."""
    dk = q.size()[-1]

    if dist_matrix is not None:
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dist_matrix_rescaled = rescale_distance_matrix(dist_matrix)
        scaled_attention_logits = matmul_qk / math.sqrt(dk) + dist_matrix_rescaled

        if mask is not None:
            scaled_attention_logits += mask * -1e9

    else:
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        if adjoin_matrix is not None:
            scaled_attention_logits += adjoin_matrix

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MOTIFAttentionAdditiveDistance(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MOTIFAttentionAdditiveDistance, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = q.size(0)

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention_additive_dist(
            q, k, v, mask, adjoin_matrix, dist_matrix
        )

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)

        return output, attention_weights


class MOTIFLayerAdditiveDistance(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate):
        super(MOTIFLayerAdditiveDistance, self).__init__()
        self.multi_head_att_1 = MOTIFAttentionAdditiveDistance(d_model // 2, num_heads)
        self.multi_head_att_2 = MOTIFAttentionAdditiveDistance(d_model // 2, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = torch.chunk(x, 2, dim=-1)

        x_l, attention_weights_local = self.multi_head_att_1(
            x1, x1, x1, encoder_padding_mask, adjoin_matrix, dist_matrix=None
        )

        x_g, attention_weights_global = self.multi_head_att_2(
            x2, x2, x2, encoder_padding_mask, adjoin_matrix=None, dist_matrix=dist_matrix,
        )

        attn_output = torch.cat([x_l, x_g], dim=-1)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2, attention_weights_local, attention_weights_global


class MOTIFAdditiveDistance(nn.Module):
    def __init__(self, input_vocab_size, d_model, dff, num_layers, num_heads, rate):
        super(MOTIFAdditiveDistance, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.dropout = nn.Dropout(rate)
        self.encoder_layers = nn.ModuleList(
            [MOTIFLayerAdditiveDistance(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )
        self.moiety_linear = nn.Linear(d_model, d_model)

    def forward(self, x, adjoin_matrix=None, dist_matrix=None):
        motif_padding_mask = create_padding_mask(x)

        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)

        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.dropout(x)
        x = self.moiety_linear(x)

        attn_local_list = []
        attn_global_list = []

        for i in range(self.num_layers):
            x, attn_l, attn_g = self.encoder_layers[i](
                x, motif_padding_mask, adjoin_matrix, dist_matrix=dist_matrix,
            )
            attn_local_list.append(attn_l)
            attn_global_list.append(attn_g)

        return (
            x,
            attn_local_list,
            attn_global_list,
            motif_padding_mask,
        )
