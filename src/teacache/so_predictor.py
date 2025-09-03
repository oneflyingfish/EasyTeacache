from collections import deque
import torch
from typing import Tuple


class SOPredictor:
    def __init__(self, max_cache=2):
        self.vt_cache = deque(maxlen=max_cache)

    def store_value(
        self,
        step: int,
        input_latent: torch.Tensor,
        output_latent: torch.Tensor,
        timestep: float = 1.0,
    ):
        if len(self.vt_cache) > 0:
            if self.vt_cache[-1][0] + 1 != step:
                self.vt_cache.clear()

        self.vt_cache.append((step, (output_latent - input_latent) / timestep))

    def predict(
        self,
        input_latent: torch.Tensor,
        timestep: float = 1.0,
    ) -> Tuple[bool, torch.Tensor]:
        if len(self.vt_cache) < 1:
            return False, None
        elif len(self.vt_cache) == 1:
            return input_latent + self.vt_cache[-1][1] * timestep
        else:
            return (
                input_latent
                + (2 * self.vt_cache[-1][1] - self.vt_cache[-2][1]) * timestep
            )
