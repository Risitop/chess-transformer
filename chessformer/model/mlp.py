import torch
import torch.nn as nn


class MLP(nn.Module):
    """A simple multi-layer perceptron with LeakyReLU activations and dropout.

    Parameters
    ----------
    in_size : int
        MLP input size.
    out_size : int
        MLP output size.
    n_hidden : int
        Number of hidden layers.
    dim_hidden : int
        Hidden layer size.
    dropout_rate : float
        Dropout rate.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        n_hidden: int,
        dim_hidden: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, dim_hidden))
        self.layers.append(nn.LayerNorm(dim_hidden))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        for _ in range(n_hidden - 1):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(nn.LayerNorm(dim_hidden))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(dim_hidden, out_size))
        self.layers.append(nn.LayerNorm(out_size))
        self.layers.append(nn.LeakyReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)
