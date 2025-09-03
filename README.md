# EasyTeacache
a easy library to use teacache for speedup dit models, contain auto params solve. support different sequence length use special params to minimum loss.

# Introduction

you can use `tea-cache` to speedup your DIT models, and easily solve the parameters associated with the model weights. 

# Usage

## Speedup mode
> if you already have params, you can use this mode

```python
from EasyTeaCache import TeaCache
cache = TeaCache(
    min_skip_step=2,        # teacache can skip first step is index==1 (start from 0)
    max_skip_step=48, # teacache can skip first step is index==48 (start from 0)
    threshold=0.04,
    model_keys=["mymodel","function"],  # any strings to sign your model-weight, support any depth
    cache_path="config/teacache/cache.json", # load config from here 
)

# in transformer.forward
skip_blocks = False
if teacache is not None:
    skip_blocks = teacache.check(
        step=time_stemp_index,
        t_mod=timestep_proj,
        sequence_length=hidden_states.size(1),
    )

if skip_blocks:
    hidden_states = teacache.update(timestep_proj, hidden_states)
else:
    input_hidden_states = hidden_states
    # 4. Transformer blocks
    for block in self.blocks:
        hidden_states = block(
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            encoder_hidden_states_mask,
        )

    if teacache is not None:
        teacache.store_truth(
            step=time_stemp_index,
            t_mod=timestep_proj,
            input_latent=input_hidden_states,
            output_latent=hidden_states,
            sequence_length=hidden_states.size(1),
        )
```

## Solve Params
> you can not use sp-parallel in this mode

```python
from EasyTeaCache import TeaCache
cache = TeaCache(
    min_skip_step=0,
    max_skip_step=-1, 
    threshold=0.04,
    model_keys=["mymodel","function"],  # any strings to sign your model-weight, support any depth
    cache_path="config/teacache/cache.json",   # save config here 
    speedup_mode=False,
)

# in transformer.forward, same with speedup mode
skip_blocks = False
if teacache is not None:
    skip_blocks = teacache.check(
        step=time_stemp_index,
        t_mod=timestep_proj,
        sequence_length=hidden_states.size(1),
    )

if skip_blocks:
    hidden_states = teacache.update(timestep_proj, hidden_states)
else:
    input_hidden_states = hidden_states
    # 4. Transformer blocks
    for block in self.blocks:
        hidden_states = block(
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            encoder_hidden_states_mask,
        )

    if teacache is not None:
        teacache.store_truth(
            step=time_stemp_index,
            t_mod=timestep_proj,
            input_latent=input_hidden_states,
            output_latent=hidden_states,
            sequence_length=hidden_states.size(1),
        )
```

# Reference

> During the implementation of this project, we referred to the methods in the papers as well as possible open-source projects (either officially released or by third parties). However, there might still be some modifications. This could be due to the unclear or non-public nature of some methods, or because I considered them to be effective improvements.

* TeaCache: https://arxiv.org/html/2411.19108v2
* FoCa: https://arxiv.org/html/2508.16211v1