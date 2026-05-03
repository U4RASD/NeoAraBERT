import os
import unicodedata

from typing import Any, Optional, Tuple

import torch
from transformers import DataCollatorForLanguageModeling, DefaultDataCollator


# Fallback POS-tag masking tables used when the user does not supply a
# `pos_masking` block in the config. Values are group-level Bernoulli
# probabilities in [0, 1]. The authoritative values live in
# `conf/neoarabert.yaml` under `datacollator.pos_masking`; these are
# only used if that block is absent.
#
# The mask linear-decay schedule interpolates per-tag between `initial`
# (start of training) and `final` (end) as:
#     P(t) = P_i + (t / T) * (P_f - P_i)
# where t is the current collator step and T is the total. `base` is
# used when `mask_linear_decay=False` or when a tag is missing from
# `initial`/`final`.
_POS_DEFAULT_INITIAL = {
    "NOUN": 0.525, "PROPN": 0.525, "ADJ": 0.450, "VERB": 0.450,
    "ADV": 0.300, "NUM": 0.300, "INTJ": 0.225,
    "PRON": 0.150, "ADP": 0.150, "PUNCT": 0.150, "SCONJ": 0.150,
    "CCONJ": 0.150, "PART": 0.150,
    "DET": 0.075, "AUX": 0.075, "X": 0.075,
}
_POS_DEFAULT_FINAL = {
    "NOUN": 0.2625, "PROPN": 0.2625, "ADJ": 0.225, "VERB": 0.225,
    "ADV": 0.15, "NUM": 0.15, "INTJ": 0.1125,
    "PRON": 0.075, "ADP": 0.075, "PUNCT": 0.075, "SCONJ": 0.075,
    "CCONJ": 0.075, "PART": 0.075,
    "DET": 0.0375, "AUX": 0.0375, "X": 0.0375,
}
_POS_DEFAULT_BASE = {
    "ADJ": 0.30, "ADP": 0.10, "ADV": 0.20, "AUX": 0.05,
    "CCONJ": 0.10, "DET": 0.05, "INTJ": 0.15,
    "NOUN": 0.35, "NUM": 0.20, "PART": 0.10, "PRON": 0.10,
    "PROPN": 0.35, "PUNCT": 0.10, "SCONJ": 0.10, "VERB": 0.30, "X": 0.05,
}


ARABIC_DIACRITIC_CHARS = {
    "\u064b",  # fathatan
    "\u064c",  # dammatan
    "\u064d",  # kasratan
    "\u064e",  # fatha
    "\u064f",  # damma
    "\u0650",  # kasra
    "\u0651",  # shadda
    "\u0652",  # sukun
    "\u0670",  # dagger alif
    "\u0653",  # maddah above
    "\u0654",  # hamza above
    "\u0655",  # hamza below
    "\u0656",  # subscript alef
    "\u0657",  # inverted damma
    "\u0658",  # mark
    "\u0659",
    "\u065a",
    "\u065b",
    "\u065c",
    "\u065d",
    "\u065e",
    "\u065f",
    "\u06d6",
    "\u06d7",
    "\u06d8",
    "\u06d9",
    "\u06da",
    "\u06db",
    "\u06dc",
    "\u06df",
    "\u06e0",
    "\u06e1",
    "\u06e2",
    "\u06e3",
    "\u06e4",
    "\u06e5",
    "\u06e6",
    "\u06e7",
    "\u06e8",
    "\u06ea",
    "\u06eb",
    "\u06ec",
    "\u06ed",
    "\u25cc",  # dotted circle placeholder often used with diacritics
}


def _is_diacritic_token(token: Optional[str]) -> bool:
    """
    Return True if the token consists solely of diacritic marks (including subword
    pieces). Used to merge diacritics into the preceding group so POS tags are
    inherited and WWM/+ grouping stays intact.
    """
    if token is None:
        return False
    # Remove WordPiece prefix for the check
    core = token[2:] if token.startswith("##") else token
    if not core:
        return False
    for ch in core:
        if ch in ARABIC_DIACRITIC_CHARS or unicodedata.combining(ch) > 0:
            continue
        return False
    return True


def _is_diac_tag(tag: Optional[str]) -> bool:
    """Return True if a disambig POS tag denotes a diacritic."""
    if tag is None:
        return False
    return tag.strip().lower() in {"diac", "diacritic", "dia", "diacritics"}
class CustomCollatorForMLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        *args,
        group_mask: bool = False,
        stemming_token: str = "[+]",
        targeted_masking: bool = False,
        log_masks: bool = True,
        log_dir: str = "logs/collator",
        pos_masking: Optional[dict] = None,
        mask_linear_decay: bool = False,
        mask_decay_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.group_mask = group_mask
        self.targeted_masking = targeted_masking
        self.log_masks = log_masks
        self.stemming_token = stemming_token

        # POS-tag masking probability tables. Source order: explicit
        # `pos_masking` dict from config -> module-level fallback defaults.
        pos_masking = dict(pos_masking) if pos_masking else {}
        self._pos_initial = dict(pos_masking.get("initial") or _POS_DEFAULT_INITIAL)
        self._pos_final = dict(pos_masking.get("final") or _POS_DEFAULT_FINAL)
        self._pos_base = dict(pos_masking.get("base") or _POS_DEFAULT_BASE)

        # Mask-rate linear-decay schedule state.
        self._mask_linear_decay_enabled = bool(mask_linear_decay)
        self._mask_decay_step = 0
        total_steps_val = int(mask_decay_steps) if mask_decay_steps is not None else 1
        self._mask_decay_steps = max(1, total_steps_val)

        # Flat log directory shared across ranks / workers. Single artifact:
        #   mask_samples.txt   first few batches rendered with [MASK]
        if not hasattr(CustomCollatorForMLM, "_log_dir"):
            os.makedirs(log_dir, exist_ok=True)
            CustomCollatorForMLM._log_dir = log_dir
            CustomCollatorForMLM._log_path = os.path.join(log_dir, "mask_samples.txt")
            CustomCollatorForMLM._log_count = 0
            CustomCollatorForMLM._log_limit = 1

        try:
            self.stemming_token_id = self.tokenizer.convert_tokens_to_ids(stemming_token)
        except Exception as e:
            self.stemming_token_id = None
            with open(CustomCollatorForMLM._log_path, "a", encoding="utf-8") as f:
                f.write("=== stemming_token_id_error ===\n")
                f.write(f"stemming_token={stemming_token!r} error={repr(e)}\n\n")

        # Per-batch disambiguation tags for targeted masking (set by torch_call).
        self._current_disambig = None

    # ------------------------------------------------------------------
    # Mask-rate linear-decay helpers
    # ------------------------------------------------------------------
    def _get_mask_decay_progress(self) -> float:
        """Return linear-decay progress t/(T-1) in [0, 1] for static targeted masking."""
        if not self._mask_linear_decay_enabled:
            return 0.0
        T = int(self._mask_decay_steps)
        if T <= 1:
            return 0.0
        t = min(max(int(self._mask_decay_step), 0), T - 1)
        return float(t) / float(T - 1)

    def set_mask_linear_decay(self, enabled: bool, total_steps: Optional[int] = None):
        """Enable/disable static linear decay and override total_steps."""
        self._mask_linear_decay_enabled = bool(enabled)
        if total_steps is not None:
            total_steps_int = int(total_steps)
            if total_steps_int > 0:
                self._mask_decay_steps = total_steps_int

    def set_mask_decay_step(self, current_step: int):
        """
        Explicitly set the internal step counter used for linear decay progress.

        This is useful if your trainer already tracks a global step t and you
        want P(t) = P_i + (t/T) * (P_f - P_i) exactly, instead of relying on
        the collator's own counter.
        """
        try:
            self._mask_decay_step = max(0, int(current_step))
        except Exception:
            pass

    def _get_token_groups(self, input_ids_row, specials_row, tokens_row):
        """Extract grouping logic from torch_mask_tokens for reuse."""
        groups = []
        current_group = []
        seq_len = len(input_ids_row)

        for i in range(seq_len):
            token_id = input_ids_row[i]
            token_str = tokens_row[i] if tokens_row is not None else None

            is_stemming_sep = self.stemming_token_id is not None and token_id == self.stemming_token_id
            # Treat stemming separators as boundaries (not part of any group)
            is_special = bool(specials_row[i])

            # End current group on special tokens or stemming separators
            if is_special or is_stemming_sep:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                continue

            # Diacritic tokens should join the preceding word/group and inherit its POS.
            if _is_diacritic_token(token_str):
                if current_group:
                    current_group.append(i)
                    continue
                if groups:
                    groups[-1].append(i)
                    continue
                # No previous group exists; start a new one with the diacritic.
                current_group = [i]
                continue

            starts_with_hashes = token_str is not None and token_str.startswith("##")

            if starts_with_hashes:
                if not current_group:
                    current_group = [i]
                else:
                    current_group.append(i)
            else:
                if not current_group:
                    current_group = [i]
                else:
                    groups.append(current_group)
                    current_group = [i]

        if current_group:
            groups.append(current_group)

        groups = [g for g in groups if len(g) > 0]
        return groups
    
    def _build_mask_groups(self, input_ids_row, specials_row, tokens_row, dis_list):
        """Build higher-level mask groups and tags from base groups + disambig list.
        
        This mirrors the grouping logic used in torch_mask_tokens for targeted masking.
        
        Args:
            input_ids_row (List[int]): Token ids for a single example.
            specials_row (List[bool]): Special-token mask for the same example.
            tokens_row (List[str]): Decoded tokens for the same example.
            dis_list (List[str]): POS tags (may be shorter/longer than groups).
        
        Returns:
            mask_groups (List[List[int]]): Final merged groups of token positions.
            mask_group_tags (List[str]): POS tag per mask group (aligned with mask_groups).
            groups (List[List[int]]): Base WWM/## groups from _get_token_groups.
            aligned_dis (List[str]): Disambig list aligned to len(groups).
        """
        # Base groups from tokenization (WWM/BPE-level)
        groups = self._get_token_groups(input_ids_row, specials_row, tokens_row)
        num_groups = len(groups)
        if num_groups == 0:
            return [], [], groups, list(dis_list or [])
        
        # Normalize disambig: replace DIAC tags with the most recent non-diac tag
        # so they inherit that POS; if none seen yet, fall back to 'X'.
        normalized_dis = []
        prev_tag = None
        for tag in list(dis_list or []):
            if _is_diac_tag(tag):
                normalized_dis.append(prev_tag if prev_tag is not None else "X")
            else:
                prev_tag = tag
                normalized_dis.append(tag)

        # Align disambig list to groups length (pad with 'X' or truncate)
        aligned_dis = normalized_dis
        if num_groups > len(aligned_dis):
            aligned_dis.extend(["X"] * (num_groups - len(aligned_dis)))
        elif num_groups < len(aligned_dis):
            # Preserve the last tag (often a suffix noun) by keeping the tail.
            if num_groups <= 0:
                aligned_dis = []
            else:
                aligned_dis = aligned_dis[: num_groups - 1] + [aligned_dis[-1]]
        
        mask_groups = []
        mask_group_tags = []
        
        gi = 0
        while gi < num_groups:
            base_group_indices = [gi]
            base_tag = aligned_dis[gi] if gi < len(aligned_dis) else "X"
            
            next_gi = gi + 1
            while next_gi < num_groups:
                last_idx = groups[base_group_indices[-1]][-1]
                first_idx_next = groups[next_gi][0]
                if first_idx_next != last_idx + 2:
                    break
                sep_idx = last_idx + 1
                same_tag = (aligned_dis[next_gi] == base_tag) if next_gi < len(aligned_dis) else False
                is_stemming_sep = (
                    self.stemming_token_id is not None
                    and 0 <= sep_idx < len(input_ids_row)
                    and input_ids_row[sep_idx] == self.stemming_token_id
                )
                if not (is_stemming_sep and same_tag):
                    break
                base_group_indices.append(next_gi)
                next_gi += 1
            
            merged_positions = []
            for idx in base_group_indices:
                merged_positions.extend(groups[idx])
            
            mask_groups.append(merged_positions)
            mask_group_tags.append(base_tag if base_tag is not None else "X")
            
            gi = base_group_indices[-1] + 1
        
        return mask_groups, mask_group_tags, groups, aligned_dis

    def _save_batch_shard(self, batch_examples, save_path, shard_idx, pbar):
        """Helper method to save a batch of examples as a shard."""
        from datasets import Dataset
        
        # Update progress bar with save status
        original_desc = pbar.desc
        pbar.set_description(f"{original_desc} | Saving shard {shard_idx}...")
        
        # Get all keys from first example
        all_keys = list(batch_examples[0].keys())
        
        # Build dictionary for this batch
        data_dict = {key: [example[key] for example in batch_examples] for key in all_keys}
        
        # Convert to Dataset
        batch_dataset = Dataset.from_dict(data_dict)
        
        # Save as a shard
        if shard_idx == 0:
            # First shard - save normally to create directory structure
            batch_dataset.save_to_disk(save_path)
        else:
            # Subsequent shards - append to existing dataset
            from datasets import load_from_disk, concatenate_datasets
            existing_dataset = load_from_disk(save_path)
            combined_dataset = concatenate_datasets([existing_dataset, batch_dataset])

            # Save to a temporary path first to avoid overwriting an open dataset
            tmp_path = save_path + "_tmp"
            import shutil as _shutil
            import gc as _gc
            if os.path.exists(tmp_path):
                _shutil.rmtree(tmp_path)
            combined_dataset.save_to_disk(tmp_path)

            # Release references to allow safe replacement
            del existing_dataset
            del combined_dataset
            _gc.collect()

            # Atomically replace old dataset directory
            if os.path.exists(save_path):
                _shutil.rmtree(save_path)
            os.rename(tmp_path, save_path)
        
        # Restore progress bar description
        pbar.set_description(original_desc)
            
    def torch_call(self, examples):
        """Override to log disambig info before processing if targeted_masking is enabled."""
        if self.targeted_masking:
            # Log disambig information for each example in the batch (respecting _log_limit)
            if self.log_masks and CustomCollatorForMLM._log_count < CustomCollatorForMLM._log_limit:
                with open(CustomCollatorForMLM._log_path, "a", encoding="utf-8") as f:
                    f.write("=== targeted_masking disambig info ===\n")
                    for idx, example in enumerate(examples):
                        if 'disambig' in example and 'input_ids' in example:
                            disambig_list = example['disambig']
                            input_ids = example['input_ids']
                            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                            specials = self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
                            
                            # Use the same grouping logic as targeted masking (mask_groups)
                            mask_groups, mask_group_tags, groups, aligned_dis = self._build_mask_groups(
                                input_ids, specials, tokens, disambig_list
                            )
                            f.write(f"Row {idx}:\n")
                            f.write(f"  len={len(aligned_dis)} | base_groups_len={len(groups)} | mask_groups_len={len(mask_groups)}\n")
                            f.write(f"POS (aligned to base groups): {aligned_dis}\n")
                            f.write(f"base_groups: {groups}\n")
                            base_grouped_tokens = [[tokens[i] for i in grp] for grp in groups]
                            f.write(f"base_grouped_tokens: {base_grouped_tokens}\n")
                            f.write(f"mask_groups: {mask_groups}\n")
                            f.write(f"mask_group_tags: {mask_group_tags}\n")
                            mask_grouped_tokens = [[tokens[i] for i in grp] for grp in mask_groups]
                            f.write(f"mask_grouped_tokens: {mask_grouped_tokens}\n\n")
                    f.write("\n")
                # NOTE: Don't increment _log_count here - let torch_mask_tokens do it after logging selection
            
            # Remove disambig column before calling parent
            # But keep a copy aligned with batch order for targeted masking inside torch_mask_tokens
            self._current_disambig = []
            filtered_examples = []
            for example in examples:
                self._current_disambig.append(example.get('disambig', []) if isinstance(example, dict) else [])
                filtered_example = {k: v for k, v in example.items() if k != 'disambig'}
                filtered_examples.append(filtered_example)
            examples = filtered_examples
        
        # Call parent's torch_call
        out = super().torch_call(examples)

        return out

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        - If group_mask is enabled, select indices by grouping, then apply 100% [MASK] replacement to selected indices.
        - Otherwise, fall back to independent Bernoulli sampling (100% [MASK]).
        """

        labels = inputs.clone()
        original_inputs = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # Targeted masking path: ignore other modes and implement POS-driven selection
        if self.targeted_masking:
            batch_size, _ = inputs.shape
            masked_indices = torch.zeros_like(inputs, dtype=torch.bool)

            # Prepare logging
            will_log = self.log_masks and (CustomCollatorForMLM._log_count < CustomCollatorForMLM._log_limit)
            log_lines = []
            if will_log:
                log_lines.append("=== targeted_masking ===\n")
                log_lines.append(f"mlm_probability: {self.mlm_probability}\n")
                log_lines.append(
                    "flags: "
                    f"targeted_masking={self.targeted_masking}  "
                    f"group_mask={self.group_mask}  "
                    f"stemming_token={getattr(self, 'stemming_token', None)}  "
                    f"stemming_token_id={self.stemming_token_id}\n"
                )
                log_lines.append("\n")

            # Pre-calculate linear decay progress for the whole batch (increment step once)
            batch_decay_progress = None
            if self._mask_linear_decay_enabled:
                batch_decay_progress = self._get_mask_decay_progress()
                self._mask_decay_step += 1

            for b in range(batch_size):
                input_ids_row = original_inputs[b].tolist()
                specials_row = special_tokens_mask[b].tolist()
                tokens_row = self.tokenizer.convert_ids_to_tokens(input_ids_row)

                # Original disambig list for this example (may be shorter/longer than base groups)
                dis_list = []
                if isinstance(self._current_disambig, list) and b < len(self._current_disambig):
                    dis_list = list(self._current_disambig[b] or [])

                # Build mask_groups using shared helper (same logic as logging)
                mask_groups, mask_group_tags, groups, dis_list = self._build_mask_groups(
                    input_ids_row, specials_row, tokens_row, dis_list
                )
                mask_group_count = len(mask_groups)
                if mask_group_count == 0:
                    continue

                # Mask-rate path: per-group Bernoulli draw with probabilities
                # taken from self._pos_initial / self._pos_final (interpolated by the
                # linear-decay schedule when enabled) or self._pos_base otherwise.
                decay_progress = batch_decay_progress if self._mask_linear_decay_enabled else None

                probs = []
                for tag in mask_group_tags:
                    tag_upper = (tag or "X").upper()
                    if decay_progress is not None and tag_upper in self._pos_initial:
                        p_i = float(self._pos_initial[tag_upper])
                        p_f = float(self._pos_final.get(tag_upper, p_i))
                        p = p_i + decay_progress * (p_f - p_i)
                    else:
                        p = float(self._pos_base.get(tag_upper, self.mlm_probability))
                    probs.append(max(0.0, min(1.0, p)))

                prob_tensor = torch.tensor(probs, dtype=torch.float32, device=inputs.device)
                group_sample = torch.bernoulli(prob_tensor).bool().tolist()
                final_selected_groups = [idx for idx, flag in enumerate(group_sample) if flag]

                # Update masked_indices for all tokens in selected groups
                for gi in final_selected_groups:
                    token_positions = mask_groups[gi]
                    masked_indices[b, token_positions] = True

                if will_log:
                    masked_ids = original_inputs[b].clone()
                    mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                    masked_ids[masked_indices[b]] = mask_token_id
                    masked_text = self.tokenizer.decode(masked_ids.tolist(), skip_special_tokens=False)
                    orig_text = self.tokenizer.decode(input_ids_row, skip_special_tokens=False)

                    log_lines.append(f"b[{b}]: num_base_groups={len(groups)} num_mask_groups={len(mask_groups)}\n")
                    selected_set = set(final_selected_groups)
                    masked_grouped_tokens = []
                    for gi in range(len(mask_groups)):
                        toks = [tokens_row[i] for i in mask_groups[gi]]
                        masked_grouped_tokens.append('[MASK]' if gi in selected_set else toks)
                    mask_grouped_tokens = [[tokens_row[i] for i in grp] for grp in mask_groups]
                    log_lines.append(f"  mask_grouped_tokens: {mask_grouped_tokens}\n")
                    log_lines.append(f"  mask_group_tags:     {mask_group_tags}\n")
                    log_lines.append(f"  masked_result:       {masked_grouped_tokens}\n")
                    log_lines.append(f"  selected_indices: {sorted(final_selected_groups)} (out of {len(mask_groups)} mask groups)\n")
                    log_lines.append(f"  text: {orig_text}\n")
                    log_lines.append(f"  text_masked: {masked_text}\n\n")

            if will_log:
                with open(CustomCollatorForMLM._log_path, "a", encoding="utf-8") as f:
                    for line in log_lines:
                        f.write(line)
                CustomCollatorForMLM._log_count += 1

            labels[~masked_indices] = -100
            mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            inputs[masked_indices] = mask_token_id

            # Clear per-batch disambig cache
            self._current_disambig = None
            return inputs, labels

        do_grouping = self.group_mask
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        if not do_grouping:
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()

            if self.log_masks and CustomCollatorForMLM._log_count < CustomCollatorForMLM._log_limit:
                with open(CustomCollatorForMLM._log_path, "a", encoding="utf-8") as f:
                    f.write("=== independent ===\n")
                    f.write(f"mlm_probability: {self.mlm_probability}\n")
                    f.write(f"flags: group_mask: {self.group_mask}  stemming_token_id: {self.stemming_token_id}\n\n")
                    for b in range(inputs.shape[0]):
                        ids = original_inputs[b].tolist()
                        toks = self.tokenizer.convert_ids_to_tokens(ids)
                        masked_ids = original_inputs[b].clone()
                        masked_ids[masked_indices[b]] = mask_token_id
                        masked_text = self.tokenizer.decode(masked_ids.tolist(), skip_special_tokens=False)
                        orig_text = self.tokenizer.decode(ids, skip_special_tokens=False)
                        f.write(f"text[{b}]: {orig_text}\n")
                        f.write(f"text_masked[{b}]: {masked_text}\n")
                        f.write(f"tokens[{b}]: {toks}\n\n")
                CustomCollatorForMLM._log_count += 1

            labels[~masked_indices] = -100
            inputs[masked_indices] = mask_token_id
            return inputs, labels

        batch_size, seq_len = inputs.shape
        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)

        all_groups = []

        for b in range(batch_size):
            input_ids_row = original_inputs[b].tolist()
            specials_row = special_tokens_mask[b].tolist()
            tokens_row = self.tokenizer.convert_ids_to_tokens(input_ids_row)

            # Use the extracted grouping logic
            groups = self._get_token_groups(input_ids_row, specials_row, tokens_row)
            all_groups.append(groups)

            if len(groups) == 0:
                continue

            # Enforce a GROUP-LEVEL budget (not token-level)
            group_budget = max(1, int(round(float(self.mlm_probability) * float(len(groups)))))
            perm = torch.randperm(len(groups)).tolist() if len(groups) > 0 else []

            selected_groups = 0
            for gi in perm:
                if selected_groups >= group_budget:
                    break
                group = groups[gi]
                masked_indices[b, group] = True
                selected_groups += 1

        if self.log_masks and CustomCollatorForMLM._log_count < CustomCollatorForMLM._log_limit:
            with open(CustomCollatorForMLM._log_path, "a", encoding="utf-8") as f:
                f.write("=== grouped ===\n")
                f.write(f"mlm_probability: {self.mlm_probability}\n")
                f.write(f"flags: group_mask: {self.group_mask}  stemming_token_id: {self.stemming_token_id}\n\n")
                for b in range(inputs.shape[0]):
                    ids = original_inputs[b].tolist()
                    toks = self.tokenizer.convert_ids_to_tokens(ids)
                    masked_ids = original_inputs[b].clone()
                    masked_ids[masked_indices[b]] = mask_token_id
                    masked_text = self.tokenizer.decode(masked_ids.tolist(), skip_special_tokens=False)
                    orig_text = self.tokenizer.decode(ids, skip_special_tokens=False)
                    f.write(f"text[{b}]: {orig_text}\n")
                    f.write(f"text_masked[{b}]: {masked_text}\n")
                    f.write(f"tokens[{b}]: {toks}\n")
                    if b < len(all_groups):
                        f.write(f"groups[{b}]: {all_groups[b]}\n")
                    f.write("\n")
            CustomCollatorForMLM._log_count += 1

        labels[~masked_indices] = -100
        inputs[masked_indices] = mask_token_id
        return inputs, labels

    def get_tag_vocab(self) -> dict:
        """Return a copy of the tag-to-id vocabulary."""
        return dict(self._tag2id)

    def get_pos_metrics_for_logging(self):
        """Return the current static linear-decay schedule values for wandb.

        Emits `mask_decay/progress` and one `mask_decay/eff/<TAG>` per
        configured POS tag. Cheap — pure function of the current step
        counter and the configured `pos_masking` tables. No file I/O,
        no per-rank aggregation, no corpus statistics.
        """
        metrics = {}

        if self.targeted_masking:
            progress = self._get_mask_decay_progress()
            metrics["mask_decay/progress"] = float(progress)

            tag_set = set(self._pos_initial) | set(self._pos_base)
            for tag_upper in sorted(tag_set):
                if self._mask_linear_decay_enabled and tag_upper in self._pos_initial:
                    p_i = float(self._pos_initial[tag_upper])
                    p_f = float(self._pos_final.get(tag_upper, p_i))
                    eff = p_i + progress * (p_f - p_i)
                else:
                    eff = float(self._pos_base.get(tag_upper, self.mlm_probability))
                metrics[f"mask_decay/eff/{tag_upper}"] = max(0.0, min(1.0, eff))

        return metrics


class DataCollatorWithPacking(DefaultDataCollator):
    """
    Data collator used for padding free approach, with sequence packing.
    """

    def __init__(self, sep_token_id, max_length, default_data_collator, **kwargs):
        super().__init__(**kwargs)
        self.sep_token_id = sep_token_id
        self.max_length = max_length
        self.default_data_collator = default_data_collator

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        packed_sequences = []
        current_sequence = []

        i = 0
        while i < len(features) or current_sequence:
            current_length = len(current_sequence)
            while current_length < self.max_length and i < len(features):
                seq = features[i]["input_ids"]
                i += 1

                current_sequence.extend(seq)
                current_length = len(current_sequence)

            # Truncate sequence and add to packed sequences
            if current_length >= self.max_length:
                packed_sequences.append({"input_ids": current_sequence[: self.max_length]})

            # Keep truncated end of sequence for the next packing
            current_sequence = current_sequence[self.max_length :] if current_length > self.max_length + 1 else []

        return self.default_data_collator(packed_sequences, return_tensors)


def get_collator(
    tokenizer,
    dtype: torch.dtype = torch.float32,
    mlm_probability: float = 0.15,
    pad_to_multiple_of: int = 8,
    mask_all: bool = False,
    group_mask: bool = False,
    stemming_token: str = "[+]",
    pack_sequences: bool = False,
    max_length: int = 512,
    targeted_masking: bool = False,
    log_masks: bool = False,
    log_dir: str = "logs/collator",
    pos_masking: Optional[dict] = None,
    mask_linear_decay: bool = False,
    mask_decay_steps: Optional[int] = None,
):
    # No need to apply any padding if sequences are packed
    if pack_sequences:
        pad_to_multiple_of = None

    mlm_collator = (
        CustomCollatorForMLM(
            tokenizer=tokenizer,
            return_tensors="pt",
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
            group_mask=group_mask,
            stemming_token=stemming_token,
            targeted_masking=targeted_masking,
            log_masks=log_masks,
            log_dir=log_dir,
            pos_masking=pos_masking,
            mask_linear_decay=mask_linear_decay,
            mask_decay_steps=mask_decay_steps,
        )
        if mask_all
        else DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            return_tensors="pt",
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
        )
    )

    if pack_sequences:
        collator = DataCollatorWithPacking(
            sep_token_id=tokenizer.sep_token_id,
            max_length=max_length,
            default_data_collator=mlm_collator,
        )

        def collate_fn(batch):
            batch = collator(batch)
            batch["attention_mask"] = None
            return batch
        
        # Attach the mlm_collator for trainer access
        collate_fn.mlm_collator = mlm_collator

    else:

        def collate_fn(batch):
            batch = mlm_collator(batch)
            batch["attention_mask"] = torch.where(batch["attention_mask"] == 1, float(0.0), float("-inf")).type(dtype)
            return batch

    # Attach the mlm_collator to the wrapper function so trainer can access it
    collate_fn.mlm_collator = mlm_collator
    
    return collate_fn
