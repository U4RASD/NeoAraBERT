import argparse
import logging
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple


class _DropFingerprintHashWarning(logging.Filter):
    def filter(self, record):
        return "couldn't be hashed properly" not in record.getMessage()


logging.getLogger("datasets.fingerprint").addFilter(_DropFingerprintHashWarning())

from datasets import (  # noqa: E402
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from omegaconf import OmegaConf  # noqa: E402

class DatasetSourceNotFound(RuntimeError):
    pass


_FORMAT_BY_EXT = {
    "parquet": "parquet",
    "arrow": "arrow",
    "json": "json",
    "jsonl": "json",
    "csv": "csv",
    "tsv": "csv",
    "txt": "text",
}


def _infer_format(path: Path) -> Optional[str]:
    if path.is_file():
        return _FORMAT_BY_EXT.get(path.suffix.lstrip(".").lower())
    files = sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        return None
    return _FORMAT_BY_EXT.get(files[0].suffix.lstrip(".").lower())


def _load_local(source_cfg) -> Optional[Dataset]:
    path = Path(source_cfg.path)
    if not path.exists():
        return None
    if path.is_dir() and (
        (path / "dataset_info.json").exists()
        or (path / "dataset_dict.json").exists()
    ):
        return load_from_disk(str(path))
    fmt = _infer_format(path)
    if fmt is None:
        return None
    data_files = source_cfg.get("data_files", None) or str(path)
    return load_dataset(fmt, data_files=data_files, split=source_cfg.get("split", None))


def _load_hub(source_cfg) -> Optional[Dataset]:
    try:
        return load_dataset(
            source_cfg.path,
            name=source_cfg.get("config_name", None),
            split=source_cfg.get("split", None),
            revision=source_cfg.get("revision", None),
        )
    except Exception as e:
        if type(e).__name__ in {"DatasetNotFoundError", "RepositoryNotFoundError"}:
            return None
        raise


LOADERS: List[Tuple[str, Callable[..., Optional[Dataset]]]] = [
    ("local filesystem", _load_local),
    ("HuggingFace Hub", _load_hub),
]


def _flatten(ds) -> Dataset:
    if isinstance(ds, DatasetDict):
        return concatenate_datasets(list(ds.values()))
    return ds


def load_source(source_cfg) -> Dataset:
    attempted = []
    for name, loader in LOADERS:
        ds = loader(source_cfg)
        if ds is not None:
            ds = _flatten(ds)
            text_column = source_cfg.text_column
            if text_column not in ds.column_names:
                raise ValueError(
                    f"text_column '{text_column}' not in dataset columns {ds.column_names}"
                )
            return ds
        attempted.append(name)
    raise DatasetSourceNotFound(
        f"'{source_cfg.path}' not found locally or on HuggingFace "
        f"(tried: {', '.join(attempted)})"
    )


def apply_sample(ds: Dataset, sample_cfg, seed: int) -> Dataset:
    """Apply the configured sampling slice to a dataset.

    `sample_cfg` may be None or have keys `size` (None|int) and
    `strategy` (head|tail|random). `random` is seeded by `seed` so two
    runs with the same seed select the same rows.
    """
    if sample_cfg is None:
        return ds
    size = sample_cfg.get("size") if hasattr(sample_cfg, "get") else None
    if size is None:
        return ds
    n = min(int(size), len(ds))
    if n <= 0:
        return ds
    strategy = (sample_cfg.get("strategy") or "head").lower()
    if strategy == "head":
        return ds.select(range(n))
    if strategy == "tail":
        return ds.select(range(len(ds) - n, len(ds)))
    if strategy == "random":
        return ds.shuffle(seed=int(seed)).select(range(n))
    raise ValueError(
        f"sample.strategy must be 'head', 'tail', or 'random', got {strategy!r}"
    )


_TATWEEL_RE = re.compile(r"ـ")
_ALIF_RE = re.compile(r"[آأإٱ]")
_ALIF_MAK_RE = re.compile(r"ى")
_TEH_MARB_RE = re.compile(r"ة")
_ZERO_WIDTH_RE = re.compile(r"[​-‍‎‏﻿]")

ARABIC_DIACRITICS = {
    "ً", "ٌ", "ٍ",
    "َ", "ُ", "ِ",
    "ّ", "ْ",
    "ٗ", "٘", "ٙ", "ٚ", "ٛ", "ٜ", "ٝ", "ٞ", "ٟ",
    "ؐ", "ؑ", "ؒ", "ؓ", "ؔ", "ؕ", "ؖ", "ؗ", "ؘ", "ؙ", "ؚ",
    "ۖ", "ۗ", "ۘ", "ۙ", "ۚ", "ۛ", "ۜ", "۟", "۠", "ۡ", "ۢ", "ۣ", "ۤ", "ۧ", "ۨ",
    "۪", "۫", "۬", "ۭ",
}


def normalize_arabic(text: str) -> str:
    text = _TATWEEL_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _ALIF_RE.sub("ا", text)
    text = _ALIF_MAK_RE.sub("ي", text)
    text = _TEH_MARB_RE.sub("ه", text)
    return text


def separate_diacritics(text: str, separator: str = "[+]") -> str:
    split_re = re.compile(rf"(\s+|{re.escape(separator)})")
    tokens = split_re.split(text)
    processed_tokens = []

    for token in tokens:
        if not token:
            continue
        if token.isspace() or token == separator:
            processed_tokens.append(token)
            continue
        if not any(c in ARABIC_DIACRITICS for c in token):
            processed_tokens.append(token)
            continue

        base_chars = []
        diac_groups = []

        for char in token:
            if char in ARABIC_DIACRITICS:
                if not diac_groups:
                    base_chars.append(" ")
                    diac_groups.append([])
                diac_groups[-1].append(char)
            else:
                base_chars.append(char)
                diac_groups.append([])

        base_word = "".join(base_chars)
        diac_string = []
        for group in diac_groups:
            if group:
                diac_string.append("".join(group))
            else:
                diac_string.append("◌")

        processed_tokens.append(base_word + " " + "".join(diac_string))
    return "".join(processed_tokens)


def build_preprocess(stemming_cfg) -> Callable[[str], str]:
    import fast_disambig

    separator = str(stemming_cfg.separator)
    preserve_diacritics = bool(stemming_cfg.get("preserve_diacritics", True))
    stemmer = fast_disambig.camel.Stemmer()

    def preprocess(text: str) -> str:
        stemmed = stemmer.stem(text, sep=separator, preserve_diacritics=preserve_diacritics)
        normalized = normalize_arabic(stemmed)
        return separate_diacritics(normalized, separator=separator)

    return preprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the flat NeoAraBERT yaml config.")
    parser.add_argument(
        "--dataset-key",
        required=True,
        choices=["main_tokenizer", "pretraining"],
        help="Which entry of pipeline.datasets to stem and write to disk.",
    )
    parser.add_argument("overrides", nargs="*", help="Optional dotlist overrides.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    source_cfg = cfg.pipeline.datasets[args.dataset_key]
    ds = load_source(source_cfg)
    ds = apply_sample(ds, source_cfg.get("sample"), seed=int(cfg.seed))

    text_column = source_cfg.text_column
    preprocess = build_preprocess(cfg.pipeline.stemming)
    ds = ds.map(
        lambda batch: {text_column: [preprocess(t) for t in batch[text_column]]},
        batched=True,
        desc="stem + normalize + separate_diacritics",
    )

    out_dir = Path(source_cfg.preprocessed_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"Prepared {len(ds)} rows with columns {ds.column_names} -> {out_dir}")


if __name__ == "__main__":
    main()
