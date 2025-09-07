"""Neural network architectures for PINNs."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from src.utils.net2net import widen_linear


class NeuralNet(nn.Module):
    """Fully connected neural network used in PINNs.

    Parameters:
        input_dim: Number of input features.
        hidden_dim: Width of each hidden layer.
        output_dim: Number of output features.
        num_hidden_layers: Number of hidden layers.
        activation: Activation function name (`tanh`, `relu`, ...).
        device: Device on which tensors should be allocated.
        dtype: Default tensor data type.
        state_dict_path: Optional path to a checkpoint for initialization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        activation: str = "tanh",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        state_dict_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(num_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        activations = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "softplus": F.softplus,
        }
        if activation not in activations:
            supported_activations = list(activations.keys())
            raise ValueError(
                f"Unsupported activation function: '{activation}'. "
                f"Supported activations are: {supported_activations}"
            )
        self.activation = activations[activation]

        # Default loss function used by training strategies
        self.mse_loss = nn.MSELoss()

        self.to(device=device, dtype=dtype)

        if state_dict_path is not None:
            self.__initialize_from_checkpoint(state_dict_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute network outputs for the given input coordinates."""
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        return self.output_layer(x)

    def __initialize_from_checkpoint(self, path: str) -> None:
        """
        Load weights from a checkpoint and perform function-preserving *widening* per layer if needed.

        Rules:
          - Load matching-shape tensors first.
          - If a layer's new width is smaller than the checkpoint width, raise:
                function-preserving shrinking is not generally possible.
          - If widths match, do nothing (already loaded).
          - If the new width is larger, copy the leading block then widen by duplication/splitting.
        """
        checkpoint = torch.load(path, map_location=self.device)
        ckpt_sd = checkpoint.get("model_state", checkpoint)
        print(f"Initializing network from checkpoint: {path}")

        # 1) Load only compatible tensors (avoids size-mismatch).
        current_sd = self.state_dict()
        compatible_sd = {k: v for k, v in ckpt_sd.items()
                         if k in current_sd and current_sd[k].shape == v.shape}
        self.load_state_dict(compatible_sd, strict=False)

        # 2) Walk adjacent pairs and handle width changes
        layers = [self.input_layer] + \
            list(self.hidden_layers) + [self.output_layer]
        names = ["input_layer"] + [f"hidden_layers.{i}" for i in range(
            len(self.hidden_layers))] + ["output_layer"]

        with torch.no_grad():
            for i in range(len(layers) - 1):
                name, next_name = names[i], names[i + 1]

                W_old = ckpt_sd.get(f"{name}.weight")
                b_old = ckpt_sd.get(f"{name}.bias")
                Wnext_old = ckpt_sd.get(f"{next_name}.weight")
                bnext_old = ckpt_sd.get(f"{next_name}.bias")

                layer, next_layer = layers[i], layers[i + 1]
                W_new = layer.weight
                Wnext_new = next_layer.weight

                h_old, h_new = W_old.shape[0], W_new.shape[0]
                add_units = h_new - h_old
                if add_units < 0:
                    raise ValueError(
                        f"Layer '{name}': new width {h_new} < old width {h_old}. "
                        "Function-preserving shrinking is not supported."
                    )

                # Copy overlap for L (rows) and its inputs (cols)
                in_cols = min(W_new.shape[1], W_old.shape[1])
                W_new[:min(h_old, h_new), :in_cols].copy_(
                    W_old[:min(h_old, h_new), :in_cols])
                if b_old is not None and layer.bias is not None:
                    layer.bias[:min(h_old, h_new)].copy_(
                        b_old[:min(h_old, h_new)])

                # Copy overlap for Lnext across BOTH rows and columns
                # handles changed out_features in hidden layers
                rows = min(Wnext_new.shape[0], Wnext_old.shape[0])
                cols = min(Wnext_new.shape[1], Wnext_old.shape[1], h_old)
                Wnext_new[:rows, :cols].copy_(Wnext_old[:rows, :cols])
                if bnext_old is not None and next_layer.bias is not None:
                    next_layer.bias[:rows].copy_(bnext_old[:rows])

                # Adjacency must hold for widening
                assert Wnext_new.shape[1] == W_new.shape[0], \
                    f"Adjacency mismatch: {next_name}.in_features={Wnext_new.shape[1]} vs {name}.out_features={W_new.shape[0]}"

                # function-preserving widening by duplication/splitting
                widen_linear(layer, next_layer, add_units)
