import torch
from torch.utils.data import DataLoader

# HuggingFace
from transformers import PreTrainedTokenizer
from datasets import Dataset

# Ours
from ..collator import get_collator


def get_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    dtype: torch.dtype = torch.float32,
    mlm_probability: float = 0.15,
    mask_all: bool = False,
    group_mask: bool = False,
    stemming_token: str = "[+]",
    pad_to_multiple_of: int = 8,
    num_workers: int = 4,
    batch_size: int = 64,
    shuffle: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    pack_sequences: bool = False,
    max_length: int = 512,
    targeted_masking: bool = False,
    log_masks: bool = True,
    log_dir: str = "logs/collator",
    pos_masking: dict = None,
    mask_linear_decay: bool = False,
    mask_decay_steps: int = 0,
) -> torch.utils.data.DataLoader:
    """Wrapper for constructing a ``torch`` dataloader, with a collator function applying masked language modeling and returning an additive pad mask.

    Args:
        dataset (Dataset).
        tokenizer (PreTrainedTokenizer).
        dtype (torch.dtype, optional): Dtype of the pad_mask. Defaults to torch.float32.
        mlm_probability (float, optional): Ratio of tokens that are masked. Defaults to 0.15.
        mask_all (bool, optional): Whether to mask every randomly selected tokens or to use the 80/10/10 masking scheme.
        group_mask (bool, optional): Enable group-level masking. Defaults to False.
        stemming_token (str, optional): Token used to separate morphemes (e.g., "[+]"). Defaults to "[+]".
        pad_to_multiple_of (int, optional): Pad to a multiple of. Defaults to 8.
        num_workers (int): Number of workers for the dataloader. Defaults to 4.
        batch_size (int): Batch size for each GPU. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the dataset at the beginning of every epoch. Defaults to True.
        pin_memory (bool, optional): If True, the dataloader will copy Tensors into device/CUDA pinned memory before returning them. Defaults to False.
        persistent_workers (bool, optional): If True, the dataloader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. Defaults to True.

    Returns:
        torch.utils.data.DataLoader
    """

    collate_fn = get_collator(
        dtype=dtype,
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=pad_to_multiple_of,
        mask_all=mask_all,
        group_mask=group_mask,
        stemming_token=stemming_token,
        max_length=max_length,
        pack_sequences=pack_sequences,
        targeted_masking=targeted_masking,
        log_masks=log_masks,
        log_dir=log_dir,
        pos_masking=pos_masking,
        mask_linear_decay=mask_linear_decay,
        mask_decay_steps=mask_decay_steps or None,
    )

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return dataloader
