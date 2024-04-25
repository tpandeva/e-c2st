import torch
import torch.nn as nn
from omegaconf import ListConfig

class CNN(nn.Module):
    """
    A multi-layer perceptron (MLP) with ReLU activation function and optional batch normalization and dropout layers.
    """

    def __init__(self,  hidden_layer_size, bias=False, nc=1):
        super(CNN, self).__init__()

        self.nc=nc
        ndf = hidden_layer_size
        self.model =  nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 2, 1, bias=bias),
            nn.GroupNorm(1,ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=bias),
            nn.GroupNorm(1,ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=bias),
            nn.GroupNorm(1,ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 3, 2, 1, bias=bias),
            nn.Flatten(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4,1, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Processed tensor.
        """
        if len(x.shape)==2: x = x.view(-1,self.nc,32, 32)
        return self.model(x)
    
    

