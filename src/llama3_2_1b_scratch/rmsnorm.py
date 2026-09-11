import torch
from torch import nn


# class RMSNorm(nn.Module):
#     def __init__(self, d_model: int, eps: float = 1e-5):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(d_model))
#         self.eps = eps

#     def forward(self, x: torch.Tensor):
#         mean_x_sq = torch.mean(x ** 2, dim=-1, keepdim=True)
#         return x * torch.rsqrt(mean_x_sq + self.eps) * self.gamma


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