'''
still have some problem, not product ready
'''

import numpy as np
import torch
from typing import Tuple, List, Dict
from collections import deque


class FDBTeaCache:
    def __init__(
        self,
        block_count: int,
        max_skip_step: int,
        min_skip_step: int = 0,
        fn: int = 1,
        bn: int = 2,
        max_consecutive_block_skip: int = -1,
        threshold_step: float = 0.04,
        enable: bool = True,
    ):
        self.threshold_step = threshold_step
        self.block_count = block_count
        self.max_skip_step = max_skip_step
        self.min_skip_step = min_skip_step
        self.fn = fn
        self.bn = bn
        self.max_consecutive_block_skip = (
            min(self.block_count - self.fn - self.bn, max_consecutive_block_skip)
            if max_consecutive_block_skip >= 0
            else (self.block_count - self.fn - self.bn)
        )

        self.enable = enable

        assert self.min_skip_block_id >= 1, "fn must in [1,block_count)"

        self.skip_blocks_sum = 0
        self.calc_blocks_sum = 0

        # cache
        self.cache_idx = None  # type: Tuple[int,int]
        self.residual_latent_per_block = None
        self.l1_distance_per_block = None

    @property
    def min_skip_block_id(self):
        return self.fn

    @property
    def max_skip_block_id(self):
        return self.block_count - self.bn

    @property
    def rel_l1_threshold(self):
        return (
            self.threshold_step / self.block_count
        ) * self.max_consecutive_block_skip

    def current_speedup_rate(self) -> float:
        if self.skip_blocks_sum + self.calc_blocks_sum < 1:
            return 1
        elif self.calc_blocks_sum < 1:
            return float("inf")
        else:
            return (self.skip_blocks_sum + self.calc_blocks_sum) / self.calc_blocks_sum

    def reset_speedup_analysis(self):
        self.skip_blocks_sum = 0
        self.calc_blocks_sum = 0

    def do_speed(self):
        return self.threshold_step > 1e-5 and self.enable

    def set_range(
        self,
        block_count: int,
        max_skip_step: int,
        min_skip_step: int = 0,
        fn: int = None,
        bn: int = None,
        max_consecutive_block_skip: int = None,
        threshold_step: float = None,
    ):
        self.block_count = block_count
        self.max_skip_step = max_skip_step
        self.min_skip_step = min_skip_step

        if fn is not None:
            self.fn = fn

        if bn is not None:
            self.bn = bn

        if max_consecutive_block_skip is not None:
            self.max_consecutive_block_skip = (
                min(self.block_count - self.fn - self.bn, max_consecutive_block_skip)
                if max_consecutive_block_skip >= 0
                else (self.block_count - self.fn - self.bn)
            )

        if threshold_step is not None:
            self.threshold_step = threshold_step

    def check(self, step: int, block_id: int) -> bool:
        """
        check skip current block or not
        """
        if not self.do_speed():
            return False

        # check step valid
        if (
            step < self.min_skip_step
            or step > self.max_skip_step
            or block_id < self.min_skip_block_id
            or block_id > self.max_skip_block_id
            or None
            in (
                self.cache_idx,
                self.residual_latent_per_block,
                self.l1_distance_per_block,
            )
        ):
            return False

        cache_step, cache_block_id = self.cache_idx
        if cache_step != step or cache_block_id >= block_id:
            # bad cache or outside
            return False

        if self.max_consecutive_block_skip > 0:
            # max skip steps
            if (block_id - cache_block_id) > self.max_consecutive_block_skip:
                return False

        if (
            (1 + self.l1_distance_per_block) ** (block_id - cache_block_id) - 1
        ) >= self.rel_l1_threshold:
            return True
        else:
            return False

    def store_truth(
        self,
        step: int,
        block_id: int,
        input_latent: torch.Tensor,
        output_latent: torch.Tensor,
    ):
        if not self.do_speed():
            return

        if step == 0 and block_id == 0:
            self.reset_speedup_analysis()
        self.calc_blocks_sum += 1

        self.cache_idx = (step, block_id)
        self.residual_latent_per_block = output_latent - input_latent
        self.l1_distance_per_block = (
            self.residual_latent_per_block.abs().mean() / input_latent.abs().mean()
        )

    def clear_cache(self):
        self.cache_idx = None  # type: Tuple[int,int]
        self.residual_latent_per_block = None
        self.l1_distance_per_block = None

    def update(
        self,
        input_latent: torch.Tensor,
    ):
        if not self.do_speed():
            return input_latent
        self.skip_blocks_sum += 1
        return self.residual_latent_per_block + input_latent
