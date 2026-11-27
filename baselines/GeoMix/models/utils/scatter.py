import torch

def scatter_batch(feature, sample_indices, batch_size, seq_len=8):
    device = feature.device
    hidden_dim = feature.size(-1)

    scattered = torch.zeros(batch_size, seq_len, hidden_dim, device=device) 

    curr_data_idx = -1
    curr_graph_idx = -1
    for graph_idx, data_idx in enumerate(sample_indices):
        if data_idx != curr_data_idx:
            curr_data_idx = data_idx
            curr_graph_idx = 0
        else:
            curr_graph_idx += 1
            try:
                assert curr_graph_idx < seq_len
            except AssertionError as e:
                print(f"Seq_len overflow! {e}")
        scattered[curr_data_idx, curr_graph_idx, :] = feature[graph_idx]
    
    return scattered