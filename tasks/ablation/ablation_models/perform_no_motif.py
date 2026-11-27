# This ablation replaces MOTIF with token embeddings and a projection.
# It omits the molecule MLP/LayerNorm and uses unmasked sums and additive amount encoding.

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
MolarRatioEncoder = perform_components.MolarRatioEncoder
TemperatureEncoder = perform_components.TemperatureEncoder
FormulationAttention = perform_components.FormulationAttention
GlobalPooling = perform_components.GlobalPooling


class PerFormNoMOTIF(nn.Module):
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
        super(PerFormNoMOTIF, self).__init__()

        self.max_num_molecules = max_num_molecules

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.embedding_linear = nn.Linear(d_model, d_model)
        self.embedding_dropout = nn.Dropout(motif_dropout)

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
        molratios = []

        for i in range(1, self.max_num_molecules + 1):
            moiety_inputs = batch[f"x{i}"].to(device)
            mr = batch[f"mr{i}"].to(device)

            molratios.append(mr)

            x = self.embedding(moiety_inputs)
            x = self.embedding_linear(x)
            x = self.embedding_dropout(x)

            mol_embedding = torch.sum(x, dim=1)
            mol_embeddings.append(mol_embedding)

        mol_embeddings = torch.stack(mol_embeddings, dim=1)
        molratios = torch.stack(molratios, dim=1)

        conc_weights = self.molar_ratio_encoder(molratios)
        scaled_mol_embeddings = mol_embeddings + conc_weights

        molecule_mask = molratios > 0

        attended_embeddings, _ = self.molecular_attention(
            scaled_mol_embeddings, molecule_mask
        )

        pooled_representation = self.global_pooling(attended_embeddings, molecule_mask)

        temperature = batch["temperature"].to(device)
        temp_weights = self.temperature_encoder(temperature)
        pooled_representation = pooled_representation * temp_weights

        prediction = self.prediction_head(pooled_representation)
        return prediction.squeeze(-1)
