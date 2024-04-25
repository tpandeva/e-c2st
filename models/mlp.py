import torch
import torch.nn as nn
from omegaconf import ListConfig


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) with ReLU activation function and optional batch normalization and dropout layers.
    """

    def __init__(self, input_size, hidden_layer_size, output_size, layer_norm=True, drop_out=True, drop_out_p=0.3,
                 bias=False):
        super(MLP, self).__init__()

        layers = []
        in_features = input_size

        # Add hidden layers with optional batch normalization and drop out
        if isinstance(hidden_layer_size, (list, ListConfig)):
            for out_features in hidden_layer_size:
                layers.append(nn.Linear(in_features, out_features, bias=bias))
                if layer_norm:
                    layers.append(nn.LayerNorm(out_features))
                layers.append(nn.ReLU())
                if drop_out:
                    layers.append(nn.Dropout(drop_out_p))
                in_features = out_features

        # Add the output layer
        layers.append(nn.Linear(in_features, output_size, bias=bias))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Processed tensor.
        """
        if len(x.shape) > 2: x = torch.flatten(x, start_dim=1)
        return self.model(x)





