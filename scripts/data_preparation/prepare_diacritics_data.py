import argparse
import logging
import sys
import unicodedata
from pathlib import Path


class _DropFingerprintHashWarning(logging.Filter):
    def filter(self, record):
        return "couldn't be hashed properly" not in record.getMessage()


logging.getLogger("datasets.fingerprint").addFilter(_DropFingerprintHashWarning())

from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import (  # noqa: E402
    apply_sample,
    load_source,
    separate_diacritics,
)

LETTER_REPLACEMENT = "ب"
DIGIT_REPLACEMENT = "1"
DOT_CIRCLE = "◌"


def diacritics_template(text: str) -> str:
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("M") or ch == DOT_CIRCLE:
            out.append(ch)
        elif cat.startswith("L"):
            out.append(LETTER_REPLACEMENT)
        elif cat == "Nd":
            out.append(DIGIT_REPLACEMENT)
        else:
            out.append(ch)
    return "".join(out)


def build_preprocess(separator: str):
    def preprocess(text: str) -> str:
        templated = diacritics_template(text)
        return separate_diacritics(templated, separator=separator)
    return preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset-key",
        required=True,
        choices=["diacritics_tokenizer"],
        help="Which entry of pipeline.datasets to template and write to disk.",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    source_cfg = cfg.pipeline.datasets[args.dataset_key]
    ds = load_source(source_cfg)
    ds = apply_sample(ds, source_cfg.get("sample"), seed=int(cfg.seed))

    text_column = source_cfg.text_column
    separator = str(cfg.pipeline.stemming.separator)
    preprocess = build_preprocess(separator)
    ds = ds.map(
        lambda batch: {text_column: [preprocess(t) for t in batch[text_column]]},
        batched=True,
        desc="diacritics-template + separate_diacritics",
    )

    out_dir = Path(source_cfg.preprocessed_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"Prepared {len(ds)} rows with columns {ds.column_names} -> {out_dir}")


if __name__ == "__main__":
    main()
