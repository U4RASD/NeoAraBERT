import re

from transformers import PreTrainedTokenizerFast
import fast_disambig


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


def separate_diacritics(text, separator):
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


def normalize_arabic(text):
    text = _TATWEEL_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _ALIF_RE.sub("ا", text)
    text = _ALIF_MAK_RE.sub("ي", text)
    text = _TEH_MARB_RE.sub("ه", text)
    return text


class ArabicMorphTokenizer(PreTrainedTokenizerFast):
    slow_tokenizer_class = None

    def __init__(
        self,
        tokenizer_file=None,
        apply_stemming=True,
        stemming_separator="[+]",
        **kwargs,
    ):
        super().__init__(tokenizer_file=tokenizer_file, **kwargs)
        self.apply_stemming = apply_stemming
        self.stemming_separator = stemming_separator
        if self.apply_stemming:
            self.stemmer = fast_disambig.camel.Stemmer()

    def _preprocess_one(self, s, do_stem):
        if isinstance(s, (list, tuple)):
            return [self._preprocess_one(x, do_stem) for x in s]
        if do_stem:
            s = self.stemmer.stem(s, sep=self.stemming_separator, preserve_diacritics=True)
        s = normalize_arabic(s)
        s = separate_diacritics(s, self.stemming_separator)
        return s

    def _preprocess_pair(self, text, text_pair, do_stem):
        def maybe(s):
            return self._preprocess_one(s, do_stem) if isinstance(s, str) else s
        if isinstance(text, (list, tuple)):
            text = [maybe(x) for x in text]
        else:
            text = maybe(text)
        if isinstance(text_pair, (list, tuple)):
            text_pair = [maybe(x) for x in text_pair]
        else:
            text_pair = maybe(text_pair)
        return text, text_pair

    def _pop_flag(self, kwargs):
        v = kwargs.pop("apply_stemming", None)
        return self.apply_stemming if v is None else bool(v)

    def __call__(self, text=None, text_pair=None, *args, **kwargs):
        flag = self._pop_flag(kwargs)
        if not getattr(self, "_processing", False):
            self._processing = True
            try:
                text, text_pair = self._preprocess_pair(text, text_pair, flag)
                return super().__call__(text=text, text_pair=text_pair, *args, **kwargs)
            finally:
                self._processing = False
        return super().__call__(text=text, text_pair=text_pair, *args, **kwargs)

    def encode(self, text, text_pair=None, *args, **kwargs):
        flag = self._pop_flag(kwargs)
        if not getattr(self, "_processing", False):
            self._processing = True
            try:
                text, text_pair = self._preprocess_pair(text, text_pair, flag)
                return super().encode(text, text_pair, *args, **kwargs)
            finally:
                self._processing = False
        return super().encode(text, text_pair, *args, **kwargs)

    def encode_plus(self, text=None, text_pair=None, *args, **kwargs):
        flag = self._pop_flag(kwargs)
        if not getattr(self, "_processing", False):
            self._processing = True
            try:
                text, text_pair = self._preprocess_pair(text, text_pair, flag)
                return super().encode_plus(text=text, text_pair=text_pair, *args, **kwargs)
            finally:
                self._processing = False
        return super().encode_plus(text=text, text_pair=text_pair, *args, **kwargs)

    def batch_encode_plus(self, batch_text_or_text_pairs=None, *args, **kwargs):
        flag = self._pop_flag(kwargs)
        if not getattr(self, "_processing", False):
            self._processing = True
            try:
                data = batch_text_or_text_pairs
                if isinstance(data, (list, tuple)):
                    new_data = []
                    for item in data:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            new_data.append(self._preprocess_pair(item[0], item[1], flag))
                        else:
                            new_data.append(self._preprocess_one(item, flag))
                    batch_text_or_text_pairs = new_data
                return super().batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs, *args, **kwargs)
            finally:
                self._processing = False
        return super().batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs, *args, **kwargs)

    def preprocess(self, text, apply_stemming=True):
        flag = self.apply_stemming if apply_stemming is None else bool(apply_stemming)
        return self._preprocess_one(text, flag)
