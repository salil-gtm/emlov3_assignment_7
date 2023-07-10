import tiktoken

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import itertools

from einops import rearrange, reduce, repeat, einsum
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning import LightningModule, LightningDataModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric

from typing import Any, Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.n_dim = n_dim
        self.h_dim = n_dim // n_heads

        self.keys = nn.Linear(n_dim, self.h_dim * self.n_heads)
        self.queries = nn.Linear(n_dim, self.h_dim * self.n_heads)
        self.values = nn.Linear(n_dim, self.h_dim * self.n_heads)

        self.proj = nn.Linear(n_dim, n_dim)

        self.layer_norm = nn.LayerNorm(n_dim)

        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        key = rearrange(
            self.keys(x), "b time (nh dim) -> nh b time dim", nh=self.n_heads
        )
        query = rearrange(
            self.queries(x), "b time (nh dim) -> nh b time dim", nh=self.n_heads
        )
        value = rearrange(
            self.values(x), "b time (nh dim) -> nh b time dim", nh=self.n_heads
        )

        energies = einsum(query, key, "nh b qt dim, nh b kt dim -> nh b qt kt")

        if mask is not None:
            fill_value = torch.finfo(energies.dtype).min
            energies = energies.masked_fill(mask, fill_value)

        attn = F.softmax(energies, dim=-1)

        attn = self.attn_dropout(attn)

        out = einsum(attn, value, "nh b qt kt, nh b kt dim -> nh b qt dim")

        out = rearrange(out, "nh b vt dim -> b vt (nh dim)")

        out = self.proj(out)

        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()

        self.fn = fn

    def forward(self, x, **kwargs):
        res = x

        out = self.fn(x, **kwargs)

        out += res

        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size=768, expansion=4, drop_p=0.0):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GPTDecoderBlock(nn.Module):
    def __init__(
        self, emb_size=768, drop_p=0.0, forward_expansion=4, forward_drop_p=0, n_heads=4
    ):
        super(GPTDecoderBlock, self).__init__()

        self.ln = nn.LayerNorm(emb_size)
        self.mha = MultiHeadAttention(n_heads=n_heads, n_dim=emb_size, dropout=drop_p)
        self.drop = nn.Dropout(drop_p)

        self.out_block = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                ),
                nn.Dropout(drop_p),
            )
        )

    def forward(self, x, mask=None):
        residual = x

        out = self.ln(x)
        out = self.mha(out, mask)
        out = self.drop(out)
        out = x + out
        out = self.out_block(out)

        return out


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_embed,
        n_heads,
        drop_p,
        n_decoder_blocks,
    ):
        super(GPT, self).__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList(
            [
                GPTDecoderBlock(
                    emb_size=n_embed, n_heads=n_heads, drop_p=drop_p
                )
            ]
            * n_decoder_blocks
        )
        self.ln = nn.LayerNorm(n_embed)
        self.ffwd = FeedForwardBlock(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # query: what am i looking for?
        # key: what do i contain?

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class GPTLitModule(LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        n_embed=64,
        block_size=8,
        n_heads=4,
        drop_p=0.0,
        n_vocab=100277,
        n_decoder_blocks=4,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # ignoring net as the model weights themselves are not a hyperparam
        self.save_hyperparameters(logger=False, ignore=["model"])

        self.learning_rate = learning_rate

        self.model = GPT(
            vocab_size=self.hparams.n_vocab,
            block_size=self.hparams.block_size,
            n_embed=self.hparams.n_embed,
            n_heads=self.hparams.n_heads,
            drop_p=self.hparams.drop_p,
            n_decoder_blocks=self.hparams.n_decoder_blocks,
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)) == 0
        )

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        mask = self.mask if targets is not None else None
        return self.model(x, targets=targets, mask=mask)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits, loss = self.forward(x, targets=y)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)

        return {"optimizer": optimizer}
