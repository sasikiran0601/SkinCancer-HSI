"""
Federated Averaging (FedAvg) simulation.

Only averages trainable parameters (``named_parameters``);
BatchNorm buffers (running_mean, running_var, num_batches_tracked)
are kept per-client and NOT averaged — avoids the well-known
FedBN instability.
"""
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from .utils import require_cuda


def federated_averaging(global_model, client_datasets, rounds=10, local_epochs=5,
                        batch_size=64, lr=1e-3, device="cuda"):
    """
    Simulate FedAvg with multiple client partitions.

    Parameters
    ----------
    global_model : nn.Module
    client_datasets : list of Dataset
        One per simulated client.
    """
    device = require_cuda(device)
    global_model.to(device)
    # Only average trainable parameters, not buffers
    param_names = [name for name, _ in global_model.named_parameters()]

    for r in range(rounds):
        local_models = []
        for client_data in client_datasets:
            local_model = copy.deepcopy(global_model)
            local_model.to(device)
            optimizer = optim.AdamW(local_model.parameters(), lr=lr)
            loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
            for _ in range(local_epochs):
                for batch_data in loader:
                    x, y = batch_data[0].to(device), batch_data[1].to(device)
                    optimizer.zero_grad()
                    out = local_model(x)
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()
            local_models.append(local_model)

        # Average trainable parameters only
        global_dict = global_model.state_dict()
        for name in param_names:
            stacked = torch.stack([lm.state_dict()[name].float() for lm in local_models])
            global_dict[name] = stacked.mean(dim=0)
        global_model.load_state_dict(global_dict)

        print(f"Federated round {r + 1}/{rounds} completed.")

    return global_model
