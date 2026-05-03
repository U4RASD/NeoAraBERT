import os
from functools import partial
from typing import Tuple

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


def get_tokenizer(
    pretrained_model_name_or_path: str,
    vocab_size: int = 32064,
    max_length: int = 4096,
    token: str = None,
    **kwargs,
):
    # Load Tokenizer and replace/add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        max_length=max_length,
        vocab_size=vocab_size,
        token=token,
        trust_remote_code=True,
    )

    # The tokenizer loaded from disk is assumed to already contain the correct
    # special-token setup (bos/eos/unk/cls/sep/pad/mask).  No further mutation
    # is performed here to avoid accidental duplication or ID-shift.
    return tokenizer


def single_column_mapping(x, tokenizer, column_name, max_length, truncation):
    return tokenizer(
        x[column_name],
        truncation=truncation,
        max_length=max_length,
        padding=False,  # no padding saves time and memory
        return_token_type_ids=False,
    )


def multi_column_mapping(x, tokenizer, column_name, max_length, truncation):
    output = {}
    for col in column_name:
        if isinstance(x[col][0], list):
            tokenized_list = [
                tokenizer(
                    item,
                    truncation=truncation,
                    max_length=max_length,
                    padding=False,
                    return_token_type_ids=False,
                    is_split_into_words=False,
                )
                for item in x[col]
            ]
            output[f"input_ids_{col}"] = [tokenized["input_ids"] for tokenized in tokenized_list]
            output[f"attention_mask_{col}"] = [tokenized["attention_mask"] for tokenized in tokenized_list]
        else:
            tokenized = tokenizer(
                x[col],
                truncation=truncation,
                max_length=max_length,
                padding=False,
                return_token_type_ids=False,
                is_split_into_words=False,
            )
            output[f"input_ids_{col}"] = tokenized["input_ids"]
            output[f"attention_mask_{col}"] = tokenized["attention_mask"]
    return output


def tokenize(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    column_name: str | Tuple[str],
    max_length: int = 4096,
    remove_columns: bool = True,
    truncation: bool = True,
    keep_columns: list | tuple = (),
    **kwargs,
):
    # Get the number of cpu cores available to the process
    num_proc = len(os.sched_getaffinity(0))

    # Remove all columns except for the `input_ids` / `attention_mask` output
    # plus anything the caller explicitly asked to preserve (e.g. POS tags).
    if remove_columns:
        columns_to_remove = [c for c in dataset.column_names if c not in keep_columns]
    else:
        columns_to_remove = None

    if isinstance(column_name, str):
        mapping = partial(single_column_mapping, tokenizer=tokenizer, column_name=column_name, max_length=max_length, truncation=truncation)
    else:
        mapping = partial(multi_column_mapping, tokenizer=tokenizer, column_name=column_name, max_length=max_length, truncation=truncation)

    # Tokenize the dataset; datasets.map infers the output schema, which
    # lets `keep_columns` entries flow through with whatever dtype they
    # already had in the source dataset.
    dataset = dataset.map(
        mapping,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns_to_remove,
    )

    return dataset
