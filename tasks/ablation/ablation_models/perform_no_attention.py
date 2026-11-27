# This ablation replaces formulation attention with a mean over all molecule slots.
# It uses slot-specific transforms, unmasked moiety sums and additive amount encoding.

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
MOTIF = perform_components.MOTIF
MolarRatioEncoder = perform_components.MolarRatioEncoder
TemperatureEncoder = perform_components.TemperatureEncoder
GlobalPooling = perform_components.GlobalPooling


class PerFormNoAttention(nn.Module):
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
        super(PerFormNoAttention, self).__init__()

        self.max_num_molecules = max_num_molecules

        self.motif = MOTIF(
            input_vocab_size=input_vocab_size,
            d_model=d_model,
            dff=d_model,
            num_layers=motif_num_layers,
            num_heads=motif_num_heads,
            rate=motif_dropout,
        )

        self.transf_layer = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(self.max_num_molecules)]
        )
        self.transf_norm = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(self.max_num_molecules)]
        )

        self.molar_ratio_encoder = MolarRatioEncoder(d_model)
        self.temperature_encoder = TemperatureEncoder(d_model)

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
        molratios = []

        for i in range(1, self.max_num_molecules + 1):
            moiety_inputs = batch[f"x{i}"].to(device)
            moiety_adj_inputs = batch[f"adjoin_matrix{i}"].to(device)
            moiety_dist_inputs = batch[f"dist_matrix{i}"].to(device)
            mr = batch[f"mr{i}"].to(device)

            molratios.append(mr)

            moiety_embeddings, *_ = self.motif(
                moiety_inputs,
                adjoin_matrix=moiety_adj_inputs,
                dist_matrix=moiety_dist_inputs,
            )

            transformed = self.transf_layer[i - 1](moiety_embeddings)
            mol_embedding = self.transf_norm[i - 1](transformed + moiety_embeddings)
            mol_embedding = torch.sum(mol_embedding, dim=1)
            mol_embeddings.append(mol_embedding)

        mol_embeddings = torch.stack(mol_embeddings, dim=1)
        molratios = torch.stack(molratios, dim=1)

        conc_weights = self.molar_ratio_encoder(molratios)
        scaled_mol_embeddings = mol_embeddings + conc_weights

        pooled_representation = torch.mean(scaled_mol_embeddings, dim=1)

        temperature = batch["temperature"].to(device)
        temp_weights = self.temperature_encoder(temperature)
        pooled_representation = pooled_representation * temp_weights

        prediction = self.prediction_head(pooled_representation)
        return prediction.squeeze(-1)
