import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def main_stemmed_dir(cfg) -> Path:
    return Path(cfg.pipeline.datasets.main_tokenizer.preprocessed_dir)


def diacritics_dir(cfg) -> Path:
    return Path(cfg.pipeline.datasets.diacritics_tokenizer.preprocessed_dir)


def pretraining_stemmed_dir(cfg) -> Path:
    return Path(cfg.pipeline.datasets.pretraining.preprocessed_dir)


def vocab_path(tokenizer_dir: Path) -> Path:
    return tokenizer_dir / "vocab.txt"


def run(cmd: list[str]) -> None:
    print("[run-pipeline]", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def stage(name: str, output: Path, force: bool) -> bool:
    if output.exists() and not force:
        print(f"[run-pipeline] skip {name}: {output} exists (use --force to redo)")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = OmegaConf.load(str(config_path))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    py = sys.executable
    overrides = list(args.overrides)

    main_stemmed = main_stemmed_dir(cfg)
    diacritics = diacritics_dir(cfg)
    pretraining_stemmed = pretraining_stemmed_dir(cfg)
    main_tok = Path(cfg.pipeline.tokenizer_main.output_dir)
    diac_tok = Path(cfg.pipeline.tokenizer_diacritics.output_dir)
    merged_tok = Path(cfg.pipeline.tokenizer_merged.output_dir)
    tokenized = Path(cfg.dataset.path_to_disk)

    if stage("prepare_main_data", main_stemmed, args.force):
        run([py, str(SCRIPTS / "data_preparation/prepare_data.py"),
             "--config", str(config_path), "--dataset-key", "main_tokenizer", *overrides])

    if stage("prepare_pretraining_data", pretraining_stemmed, args.force):
        run([py, str(SCRIPTS / "data_preparation/prepare_data.py"),
             "--config", str(config_path), "--dataset-key", "pretraining", *overrides])

    if stage("prepare_diacritics_data", diacritics, args.force):
        run([py, str(SCRIPTS / "data_preparation/prepare_diacritics_data.py"),
             "--config", str(config_path), "--dataset-key", "diacritics_tokenizer", *overrides])

    if stage("train_main_tokenizer", vocab_path(main_tok), args.force):
        run([py, str(SCRIPTS / "tokenizer_training/train_wordpiece_tokenizer.py"),
             "--data", str(main_stemmed),
             "--column", str(cfg.pipeline.datasets.main_tokenizer.text_column),
             "--output", str(main_tok),
             "--vocab_size", str(cfg.pipeline.tokenizer_main.vocab_size),
             "--num_unused_tokens", str(cfg.pipeline.tokenizer_main.num_unused_tokens),
             "--min_frequency", str(cfg.pipeline.tokenizer_main.min_frequency),
             "--stemming-separator", str(cfg.pipeline.stemming.separator)])

    if stage("train_diacritics_tokenizer", vocab_path(diac_tok), args.force):
        run([py, str(SCRIPTS / "tokenizer_training/train_wordpiece_tokenizer.py"),
             "--data", str(diacritics),
             "--column", str(cfg.pipeline.datasets.diacritics_tokenizer.text_column),
             "--output", str(diac_tok),
             "--vocab_size", str(cfg.pipeline.tokenizer_diacritics.vocab_size),
             "--num_unused_tokens", str(cfg.pipeline.tokenizer_diacritics.num_unused_tokens),
             "--min_frequency", str(cfg.pipeline.tokenizer_diacritics.min_frequency),
             "--stemming-separator", str(cfg.pipeline.stemming.separator),
             "--keep-diacritics"])

    if stage("merge_tokenizers", vocab_path(merged_tok), args.force):
        run([py, str(SCRIPTS / "tokenizer_training/merge_tokenizers.py"),
             "--base-dir", str(main_tok),
             "--diacs-dir", str(diac_tok),
             "--max-diacritics-only-tokens", str(cfg.pipeline.tokenizer_merged.max_diacritics_only_tokens),
             "--out-dir", str(merged_tok)])

    runtime_overrides = [
        f"tokenizer.pretrained_model_name_or_path={merged_tok}",
        f"dataset.train.path={pretraining_stemmed}",
        f"dataset.column={cfg.pipeline.datasets.pretraining.text_column}",
    ] + overrides

    if stage("preprocess", tokenized, args.force):
        run([py, str(SCRIPTS / "pretraining/preprocess.py"),
             "--config", str(config_path), *runtime_overrides])

    if not args.skip_pretrain:
        accelerate_cfg = OmegaConf.to_container(cfg.accelerate, resolve=True)
        with tempfile.TemporaryDirectory() as td:
            accel_cfg_path = Path(td) / "accelerate_config.yaml"
            with open(accel_cfg_path, "w") as f:
                yaml.safe_dump(accelerate_cfg, f, sort_keys=False)
            accelerate_bin = Path(py).parent / "accelerate"
            run([str(accelerate_bin), "launch", "--config_file", str(accel_cfg_path),
                 str(SCRIPTS / "pretraining/pretrain.py"),
                 "--config", str(config_path), *runtime_overrides])

    if not args.skip_export:
        run([py, str(SCRIPTS / "pretraining/export_pretrained.py"),
             "--config", str(config_path), *runtime_overrides])

    print("[run-pipeline] done")


if __name__ == "__main__":
    main()
