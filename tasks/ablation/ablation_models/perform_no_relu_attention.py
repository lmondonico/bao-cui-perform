# This ablation removes ReLU from the MOTIF global attention scores.
# Distance weighting and the remaining model connections are retained.

import torch
import torch.nn as nn
import sys
import os


sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import importlib

perform_components = importlib.import_module("modules.perform.perform_components")
MOTIFNoReLU = perform_components.MOTIFNoReLU
MolarRatioEncoder = perform_components.MolarRatioEncoder
TemperatureEncoder = perform_components.TemperatureEncoder
FormulationAttention = perform_components.FormulationAttention
GlobalPooling = perform_components.GlobalPooling


class PerFormNoReLUAttention(nn.Module):
    def __init__(
        self,
        d_model,
        motif_num_layers,
        motif_num_heads,
        motif_dropout,
        main_att_num_heads,
        main_att_dropout,
        final_dropout,
        input_vocab_size,
        max_num_molecules,
    ):
        super(PerFormNoReLUAttention, self).__init__()

        self.max_num_molecules = max_num_molecules

        self.motif = MOTIFNoReLU(
            input_vocab_size=input_vocab_size,
            d_model=d_model,
            dff=d_model,
            num_layers=motif_num_layers,
            num_heads=motif_num_heads,
            rate=motif_dropout,
        )

        self.transf_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.transf_norm = nn.LayerNorm(d_model)

        self.molar_ratio_encoder = MolarRatioEncoder(d_model)
        self.temperature_encoder = TemperatureEncoder(d_model)

        self.molecular_attention = FormulationAttention(
            d_model=d_model, num_heads=main_att_num_heads, dropout=main_att_dropout
        )

        self.global_pooling = GlobalPooling()

        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(final_dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, batch):
        device = next(self.parameters()).device
        mol_embeddings = []
        molar_ratios = []

        for i in range(1, self.max_num_molecules + 1):
            moiety_inputs = batch[f"x{i}"].to(device)
            moiety_adj_inputs = batch[f"adjoin_matrix{i}"].to(device)
            moiety_dist_inputs = batch[f"dist_matrix{i}"].to(device)
            mr = batch[f"mr{i}"].to(device)
            if mr.dim() > 1:
                mr = mr.squeeze(-1)

            molar_ratios.append(mr)

            moiety_embeddings, *_ = self.motif(
                moiety_inputs,
                adjoin_matrix=moiety_adj_inputs,
                dist_matrix=moiety_dist_inputs,
            )

            transformed = self.transf_mlp(moiety_embeddings)
            mol_embedding = self.transf_norm(transformed + moiety_embeddings)
            mask = (moiety_inputs != 0).float().unsqueeze(-1)
            mol_embedding = mol_embedding * mask
            mol_embedding = torch.sum(mol_embedding, dim=1)
            mol_embeddings.append(mol_embedding)

        mol_embeddings = torch.stack(mol_embeddings, dim=1)
        molar_ratios = torch.stack(molar_ratios, dim=1)

        mr_weights = self.molar_ratio_encoder(molar_ratios)
        scaled_mol_embeddings = mol_embeddings * mr_weights

        molecule_mask = molar_ratios > 0

        attended_embeddings, _ = self.molecular_attention(
            scaled_mol_embeddings, molecule_mask
        )

        pooled_representation = self.global_pooling(attended_embeddings, molecule_mask)

        temperature = batch["temperature"].to(device)
        temp_weights = self.temperature_encoder(temperature)
        pooled_representation = pooled_representation * temp_weights

        prediction = self.prediction_head(pooled_representation)
        return prediction.squeeze(-1)
