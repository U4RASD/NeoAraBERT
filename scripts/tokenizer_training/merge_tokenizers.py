import argparse
import json
import unicodedata
from pathlib import Path
from shutil import copy2
from typing import Dict, List, Set, Tuple


COPIED_FILES = ("tokenizer_config.json", "special_tokens_map.json")
DOT_CIRCLE = "◌"

ARABIC_DIACRITICS = {
    "ً", "ٌ", "ٍ",
    "َ", "ُ", "ِ",
    "ّ", "ْ",
    "ٗ", "٘", "ٙ", "ٚ", "ٛ", "ٜ", "ٝ", "ٞ", "ٟ",
    "ؐ", "ؑ", "ؒ", "ؓ", "ؔ", "ؕ", "ؖ", "ؗ", "ؘ", "ؙ", "ؚ",
    "ۖ", "ۗ", "ۘ", "ۙ", "ۚ", "ۛ", "ۜ", "۟", "۠", "ۡ", "ۢ", "ۣ", "ۤ", "ۧ", "ۨ",
    "۪", "۫", "۬", "ۭ",
}


def get_mandatory_tokens() -> Set[str]:
    tokens: Set[str] = set()
    for d in ARABIC_DIACRITICS:
        tokens.add(d)
        tokens.add("##" + d)
    tokens.add(DOT_CIRCLE)
    tokens.add("##" + DOT_CIRCLE)
    return tokens


def is_diacritic_only_token(token: str) -> bool:
    if not token:
        return False
    core = token[2:] if token.startswith("##") else token
    if not core:
        return False
    for ch in core:
        if ch == DOT_CIRCLE:
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        return False
    return True


def read_vocab_txt(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [x for x in lines if x != ""]


def write_vocab_txt(path: Path, tokens: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for t in tokens:
            f.write(t)
            f.write("\n")


def load_tokenizer_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_tokenizer_json(path: Path, tok: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, indent=2)


def pick_first_n_unique_matching(
    diacs_vocab_in_order: List[str],
    base_seen: Set[str],
    n: int,
) -> Tuple[List[str], int]:
    picked: List[str] = []
    picked_set: Set[str] = set()

    for m in get_mandatory_tokens():
        if m not in base_seen and m not in picked_set:
            picked.append(m)
            picked_set.add(m)

    scanned = 0
    for tok in diacs_vocab_in_order:
        scanned += 1
        if len(picked) >= n:
            break
        if not is_diacritic_only_token(tok):
            continue
        if tok in base_seen:
            continue
        if tok in picked_set:
            continue
        picked.append(tok)
        picked_set.add(tok)

    return picked, scanned


def shift_added_token_ids_if_needed(tok: Dict, old_vocab_size: int, new_vocab_size: int) -> None:
    delta = new_vocab_size - old_vocab_size
    if delta == 0:
        return
    added_tokens = tok.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        return
    for entry in added_tokens:
        if not isinstance(entry, dict):
            continue
        tid = entry.get("id")
        if isinstance(tid, int) and tid >= old_vocab_size:
            entry["id"] = tid + delta


def merge(base_dir: Path, diacs_dir: Path, max_diacritics_only_tokens: int, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    base_vocab_path = base_dir / "vocab.txt"
    diacs_vocab_path = diacs_dir / "vocab.txt"
    base_tok_path = base_dir / "tokenizer.json"

    if not base_vocab_path.exists():
        raise FileNotFoundError(f"Missing base vocab.txt: {base_vocab_path}")
    if not diacs_vocab_path.exists():
        raise FileNotFoundError(f"Missing diacs vocab.txt: {diacs_vocab_path}")
    if not base_tok_path.exists():
        raise FileNotFoundError(f"Missing base tokenizer.json: {base_tok_path}")

    base_vocab = read_vocab_txt(base_vocab_path)
    diacs_vocab_all = read_vocab_txt(diacs_vocab_path)
    target_n = max(0, max_diacritics_only_tokens)

    base_seen = set(base_vocab)
    picked, _ = pick_first_n_unique_matching(diacs_vocab_all, base_seen, target_n)
    if len(picked) < target_n:
        print(
            f"[merge] requested max_diacritics_only_tokens={target_n} but only {len(picked)} "
            f"diacritic-only tokens are available in {diacs_vocab_path}; "
            f"merging with what was found."
        )

    merged_vocab = list(base_vocab) + picked

    for fname in COPIED_FILES:
        src = base_dir / fname
        if src.exists():
            copy2(src, out_dir / fname)

    write_vocab_txt(out_dir / "vocab.txt", merged_vocab)

    tok = load_tokenizer_json(base_tok_path)
    model = tok.get("model")
    if not isinstance(model, dict) or model.get("type") != "WordPiece":
        raise ValueError("Base tokenizer.json does not look like a WordPiece tokenizer.")

    old_vocab_obj = tok["model"].get("vocab", {})
    if not isinstance(old_vocab_obj, dict):
        raise ValueError("Base tokenizer.json model.vocab is not a dict.")
    old_vocab_size = len(old_vocab_obj)
    new_vocab_size = len(merged_vocab)

    shift_added_token_ids_if_needed(tok, old_vocab_size=old_vocab_size, new_vocab_size=new_vocab_size)
    tok["model"]["vocab"] = {token: idx for idx, token in enumerate(merged_vocab)}

    save_tokenizer_json(out_dir / "tokenizer.json", tok)
    return new_vocab_size


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--diacs-dir", required=True)
    p.add_argument("--max-diacritics-only-tokens", dest="max_diacritics_only_tokens", type=int, default=5000)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    new_size = merge(
        base_dir=Path(args.base_dir),
        diacs_dir=Path(args.diacs_dir),
        max_diacritics_only_tokens=int(args.max_diacritics_only_tokens),
        out_dir=Path(args.out_dir),
    )
    print(f"Merged tokenizer vocab size: {new_size}")
    print(f"Wrote folder: {args.out_dir}")


if __name__ == "__main__":
    main()
