---
license: cc-by-4.0
language:
- code
---

### Overview

DPO dataset meant to enhance python coding abilities.

This dataset uses the excellent https://huggingface.co/datasets/Vezora/Tested-22k-Python-Alpaca dataset as the "chosen" responses, given this dataset was already tested and validated.

The "rejected" values were generated with a mix of airoboros-l2-13b-3.1 and bagel-7b-v0.1.

The rejected values may actually be perfectly fine, but the assumption here is that the values are generally a lower quality than the chosen counterpart.  Items with duplicate code blocks were removed.

### Contribute

If you're interested in new functionality/datasets, take a look at [bagel repo](https://github.com/jondurbin/bagel) and [airoboros](https://github.com/jondurbin/airoboros) and either make a PR or open an issue with details.

To help me with the fine-tuning costs, dataset generation, etc., please use one of the following:

- https://bmc.link/jondurbin
- ETH 0xce914eAFC2fe52FdceE59565Dd92c06f776fcb11
- BTC bc1qdwuth4vlg8x37ggntlxu5cjfwgmdy5zaa7pswf