import argparse
from pathlib import Path

from datasets import load_dataset, load_from_disk
from omegaconf import OmegaConf

from neoarabert.tokenizer import get_tokenizer, tokenize


def _is_save_to_disk_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "dataset_info.json").exists()
        or (path / "dataset_dict.json").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    tokenizer = get_tokenizer(**cfg.tokenizer)
    print(tokenizer)

    print("Loading dataset")
    train_cfg = cfg.dataset.train
    src_path = Path(str(train_cfg.get("path", "")))
    if _is_save_to_disk_dir(src_path):
        dataset = load_from_disk(str(src_path))
    else:
        dataset = load_dataset(**train_cfg)

    keep_columns = list(cfg.dataset.get("keep_columns", []) or [])
    print(f"Tokenizing dataset (preserving columns: {keep_columns or 'none'})")
    dataset = tokenize(
        dataset,
        tokenizer,
        column_name=cfg.dataset.column,
        keep_columns=keep_columns,
        **cfg.tokenizer,
    )

    print(f"Saving tokenized dataset to {cfg.dataset.path_to_disk}")
    dataset.save_to_disk(cfg.dataset.path_to_disk, max_shard_size="1GB")


if __name__ == "__main__":
    main()
