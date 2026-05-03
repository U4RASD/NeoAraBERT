import argparse
import os
import re

from datasets import DatasetDict, load_from_disk
from tokenizers import BertWordPieceTokenizer
from tokenizers.normalizers import Sequence
from tokenizers.processors import TemplateProcessing
from transformers import BertTokenizerFast

DIACRITICS_PATTERN = re.compile(r" [ً-ٰٟۖ-ۜ۟-ۤۧ-۪ۨ-ۭ◌]+")


def clean_text(text: str) -> str:
    return DIACRITICS_PATTERN.sub("", text)


def get_files(p: str):
    if os.path.isfile(p):
        return [p]
    return [os.path.join(p, f) for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--num_unused_tokens", type=int, default=0)
    parser.add_argument("--column", type=str, default="text")
    parser.add_argument("--keep-diacritics", dest="keep_diacritics", action="store_true",
                        help="Skip the post-space diacritic strip; needed when training a diacritics-only tokenizer.")
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--stemming-separator", dest="stemming_separator", required=True,
                        help="Stem-piece separator from pipeline.stemming.separator; baked into the vocab as a special token.")
    args = parser.parse_args()

    wp = BertWordPieceTokenizer(lowercase=False, strip_accents=False)
    wp.normalizer = Sequence([])

    special_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", args.stemming_separator,
    ] + [f"[unused{i}]" for i in range(args.num_unused_tokens)]

    transform = (lambda t: t) if args.keep_diacritics else clean_text

    is_hf_dataset = os.path.isdir(args.data) and (
        os.path.exists(os.path.join(args.data, "dataset_info.json"))
        or os.path.exists(os.path.join(args.data, "dataset_dict.json"))
    )

    if is_hf_dataset:
        ds = load_from_disk(args.data)

        def iterator():
            if isinstance(ds, DatasetDict):
                for split in ds.values():
                    for txt in split[args.column]:
                        yield transform(txt)
            else:
                for txt in ds[args.column]:
                    yield transform(txt)

        total_len = sum(len(s) for s in ds.values()) if isinstance(ds, DatasetDict) else len(ds)
        wp.train_from_iterator(
            iterator(),
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
            length=total_len,
        )
    else:
        wp.train(
            files=get_files(args.data),
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
        )

    wp.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", wp.token_to_id("[CLS]")),
            ("[SEP]", wp.token_to_id("[SEP]")),
        ],
    )

    hf_tok = BertTokenizerFast(
        tokenizer_object=wp,
        unk_token="[UNK]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        do_lower_case=False,
        additional_special_tokens=[args.stemming_separator, *[f"[unused{i}]" for i in range(args.num_unused_tokens)]],
    )
    hf_tok.save_pretrained(args.output)
    print(f"Saved tokenizer to {args.output}")


if __name__ == "__main__":
    main()
