import torch
from torch import nn
import torch.nn.functional as F


class LlamaRoPE(nn.Module):
    def __init__(self, max_seq_len=131072, d_h = 64, rho = 500000, f = 32, f_low = 1, f_high = 4, L0 = 8192):
        super().__init__()

        d_half = d_h // 2

        inv_freq = rho ** (-torch.arange(0, d_half) / d_half)
        wavelength = (2 * torch.pi) / inv_freq

        low_wavelen = L0 / f_low
        high_wavelen = L0 / f_high

        smooth_factor = (
            (L0 / wavelength) - f_low
        ) / (f_high - f_low)

        smooth_factor = torch.clamp(
            smooth_factor,
            0.0,
            1.0
        )

        inv_freq_smoothed = (
            (1 - smooth_factor) * inv_freq / f
            + smooth_factor * inv_freq
        )


        is_low = wavelength > low_wavelen
        is_high = wavelength < high_wavelen

        inv_freq_scaled = torch.where(
            is_low,
            inv_freq / f,
            torch.where(
                is_high,
                inv_freq,
                inv_freq_smoothed
            )
        )
        self.register_buffer("inv_freq", inv_freq_scaled)

        p = torch.arange(0, max_seq_len).reshape(-1,1) # position p
        phi = p * inv_freq_scaled.reshape(1,-1) # phi_i (N x d_h/2)
        cos = torch.cos(phi)
        sin = torch.sin(phi)

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, position_ids=None):
        if position_ids is None:
            L = x.shape[-2]
            cos = self.cos[:L]
            sin = self.sin[:L]
        else:
            cos = self.cos[position_ids].unsqueeze(1)
            sin = self.sin[position_ids].unsqueeze(1)

        # Llama HF uses the "rotate half" convention:
        # q0 is paired with q32, q1 with q33, ..., q31 with q63.
        # Therefore, split the head dimension into two halves
        # instead of taking even/odd indices (::2 and 1::2).
        # and coz of this, we have to cat and not use stack to createfinal RoPE
        half = x.shape[-1] // 2

        x_first = x[..., :half]
        x_second = x[..., half:]

        x_rope_first = x_first * cos - x_second * sin
        x_rope_second = x_first * sin + x_second * cos

        return torch.cat(
            [x_rope_first, x_rope_second],
            dim=-1
        )