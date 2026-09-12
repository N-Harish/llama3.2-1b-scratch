import torch


class KVCache:
    def __init__(self, initial_capacity: int = 512):
        self.capacity = initial_capacity
        self.seen_tokens = 0
        self.k = None
        self.v = None

    def _allocate(self, batch_size, n_heads, capacity, head_dim, dtype, device):
        return torch.empty(
            (batch_size, n_heads, capacity, head_dim),
            dtype=dtype,
            device=device,
        )

    def update(self, k: torch.Tensor, v: torch.Tensor):
        # k, v shape: [B, n_kv_heads, seq_len, head_dim]
        B, H, S, D = k.shape
        
        needed_capacity = self.seen_tokens + S

        # 1. First-time initialization
        if self.k is None:
            self.capacity = max(self.capacity, needed_capacity)
            self.k = self._allocate(B, H, self.capacity, D, k.dtype, k.device)
            self.v = self._allocate(B, H, self.capacity, D, v.dtype, v.device)

        # 2. Geometric growth (only triggers when capacity is breached)
        elif needed_capacity > self.capacity:
            while self.capacity < needed_capacity:
                self.capacity *= 2  # Double capacity

            new_k = self._allocate(B, H, self.capacity, D, k.dtype, k.device)
            new_v = self._allocate(B, H, self.capacity, D, v.dtype, v.device)

            # Copy existing history
            new_k[:, :, :self.seen_tokens, :] = self.k[:, :, :self.seen_tokens, :]
            new_v[:, :, :self.seen_tokens, :] = self.v[:, :, :self.seen_tokens, :]

            self.k = new_k
            self.v = new_v

        # 3. Fast in-place assignment (no allocation during normal decode steps)
        start = self.seen_tokens
        end = start + S
        self.k[:B, :, start:end, :] = k
        self.v[:B, :, start:end, :] = v
        self.seen_tokens = end

        # Return a view of valid tokens
        return self.k[:B, :, :end, :], self.v[:B, :, :end, :]

    def clear(self):
        self.seen_tokens = 0
        # Optional: release GPU memory between distinct requests
        self.k = None
        self.v = None

    def is_empty(self):
        return self.seen_tokens == 0
