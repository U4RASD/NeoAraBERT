import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf, open_dict
from transformers import AutoTokenizer

from neoarabert.exportable import model as exportable_model_pkg
from neoarabert.exportable.model import NeoAraBERTConfig, NeoAraBERTLMHead


AUTO_MAP_MODEL = {
    "AutoConfig": "model.NeoAraBERTConfig",
    "AutoModel": "model.NeoAraBERT",
    "AutoModelForMaskedLM": "model.NeoAraBERTLMHead",
    "AutoModelForSequenceClassification": "model.NeoAraBERTForSequenceClassification",
}

AUTO_MAP_TOKENIZER = {"AutoTokenizer": ["tokenizer.ArabicMorphTokenizer", None]}

EXPORT_FILES = ("model.py", "rotary.py", "tokenizer.py")

REQUIRED_SHAPE_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
)

HARDCODED_ARCH = {"rope": True, "rms_norm": True, "hidden_act": "swiglu"}


def find_latest_checkpoint(model_checkpoint_dir: Path) -> Optional[Path]:
    if not model_checkpoint_dir.is_dir():
        return None
    steps = sorted(
        (int(p.name) for p in model_checkpoint_dir.iterdir() if p.name.isdigit()),
        reverse=True,
    )
    if not steps:
        return None
    return model_checkpoint_dir / str(steps[0]) / "state_dict.pt"


def _validate_yaml(cfg) -> None:
    missing = [f for f in REQUIRED_SHAPE_FIELDS if f not in cfg.model]
    if missing:
        raise ValueError(
            f"cfg.model missing required shape fields {missing}. "
            f"These define the parameter count and cannot fall back to defaults."
        )
    for field in ("vocab_size", "max_length"):
        if field not in cfg.tokenizer:
            raise ValueError(f"cfg.tokenizer.{field} is required for export.")

    for field, expected in HARDCODED_ARCH.items():
        actual = cfg.model.get(field, None)
        if actual is None:
            continue
        if isinstance(expected, str):
            ok = str(actual).lower() == expected
        else:
            ok = actual == expected
        if not ok:
            raise ValueError(
                f"cfg.model.{field}={actual!r} disagrees with the inference "
                f"template's hardcoded {field}={expected!r}. Update "
                f"src/neoarabert/exportable/model.py to support {field}={actual!r}, "
                f"or change the yaml to match."
            )


def _config_kwargs_from_yaml(cfg) -> dict:
    """Pass through every cfg.model field plus vocab_size/max_length from
    cfg.tokenizer. Unknown fields land in kwargs and get serialized verbatim
    into config.json, so the yaml is the single source of truth."""
    kwargs = dict(OmegaConf.to_container(cfg.model, resolve=True))
    kwargs["vocab_size"] = int(cfg.tokenizer.vocab_size)
    kwargs["max_length"] = int(cfg.tokenizer.max_length)
    return kwargs


def _format_default(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return f'"{v}"'
    return repr(v)


def _patch_model_py_defaults(model_py: Path, values: dict) -> None:
    """Rewrite NeoAraBERTConfig.__init__ defaults so the on-disk model.py
    advertises the values this checkpoint was trained with, not the
    template's last-resort fallbacks. Per-parameter regex match on
    `name: type = default,`; the `: type =` shape only occurs in function
    signatures, so we don't need to scope to a particular block."""
    src = model_py.read_text(encoding="utf-8")
    for name, value in values.items():
        pattern = rf"(\b{re.escape(name)}\s*:\s*[^=\n]+=\s*)([^,\n]+)(\s*,\s*\n)"
        replacement = lambda m, v=value: m.group(1) + _format_default(v) + m.group(3)
        src = re.sub(pattern, replacement, src, count=1)
    model_py.write_text(src, encoding="utf-8")


def export(
    cfg,
    checkpoint: Path,
    out_dir: Path,
    tokenizer_dir: Path,
) -> None:
    _validate_yaml(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = NeoAraBERTConfig(**_config_kwargs_from_yaml(cfg))

    model = NeoAraBERTLMHead(config)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if state and next(iter(state)).startswith("module."):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("freqs_cis")]
    if missing:
        print(f"[export] missing keys: {missing}")
    if unexpected:
        print(f"[export] unexpected keys: {unexpected}")

    model.save_pretrained(str(out_dir), safe_serialization=True)

    template_dir = Path(exportable_model_pkg.__file__).parent
    for fname in EXPORT_FILES:
        shutil.copy2(template_dir / fname, out_dir / fname)

    _patch_model_py_defaults(out_dir / "model.py", config.to_dict())

    if tokenizer_dir.is_dir():
        for f in tokenizer_dir.iterdir():
            if f.name in {"vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}:
                shutil.copy2(f, out_dir / f.name)

    config_path = out_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg_json = json.load(f)
    cfg_json["auto_map"] = dict(AUTO_MAP_MODEL)
    cfg_json["architectures"] = ["NeoAraBERTLMHead"]
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg_json, f, ensure_ascii=False, indent=2)

    tok_cfg_path = out_dir / "tokenizer_config.json"
    if tok_cfg_path.exists():
        with tok_cfg_path.open("r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        tok_cfg["tokenizer_class"] = "ArabicMorphTokenizer"
        tok_cfg["auto_map"] = dict(AUTO_MAP_TOKENIZER)
        tok_cfg["trust_remote_code"] = True
        tok_cfg["apply_stemming"] = True
        tok_cfg["stemming_separator"] = str(cfg.pipeline.stemming.separator)
        with tok_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(tok_cfg, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        ckpt = find_latest_checkpoint(Path(cfg.trainer.dir) / "model_checkpoints")
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found ({ckpt})")

    out_dir = Path(args.out_dir or str(cfg.pipeline.export.output_dir))
    tokenizer_dir = Path(args.tokenizer_dir or str(cfg.tokenizer.pretrained_model_name_or_path))

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    with open_dict(cfg.tokenizer):
        cfg.tokenizer.vocab_size = len(tokenizer)

    export(cfg, ckpt, out_dir, tokenizer_dir)
    print(f"[export] wrote {out_dir}")


if __name__ == "__main__":
    main()
