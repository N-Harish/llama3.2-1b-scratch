import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype

        x = x.float()

        variance = x.pow(2).mean(-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.eps)

        return self.gamma * x.to(input_dtype)
