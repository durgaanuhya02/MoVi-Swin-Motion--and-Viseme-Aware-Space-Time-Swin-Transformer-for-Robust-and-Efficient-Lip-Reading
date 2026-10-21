"""Text -> phoneme -> viseme conversion shared by every stage of the pipeline.

Phonemes are the 39 ARPAbet phonemes of the CMU Pronouncing Dictionary with
stress markers removed (as in VALLR). Visemes group phonemes that look alike
on the lips; the grouping is adapted from the consonant/vowel classes compared
in Cappelletta & Harte (EUSIPCO 2011) and Bear et al. (ISVC 2014).

Index 0 of both vocabularies is the CTC blank.
"""
import re
from functools import lru_cache

PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]

VISEME_GROUPS = {
    "BILABIAL":    ["P", "B", "M"],
    "LABIODENTAL": ["F", "V"],
    "DENTAL":      ["TH", "DH"],
    "ALVEOLAR":    ["T", "D", "N", "L"],
    "SIBILANT":    ["S", "Z"],
    "POSTALVEOLAR": ["SH", "ZH", "CH", "JH"],
    "BACK":        ["K", "G", "NG", "HH"],
    "RHOTIC":      ["R", "ER"],
    "ROUND_CLOSE": ["W", "UW", "UH"],
    "SPREAD_CLOSE": ["Y", "IY", "IH"],
    "SPREAD_MID":  ["EH", "EY", "AE"],
    "OPEN":        ["AA", "AH", "AY", "AW"],
    "ROUND_OPEN":  ["AO", "OW", "OY"],
}
VISEMES = list(VISEME_GROUPS)

BLANK = "<blank>"
PHONEME_VOCAB = [BLANK] + PHONEMES                    # 40 entries
VISEME_VOCAB = [BLANK] + VISEMES                      # 14 entries
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_VOCAB)}
VISEME_TO_ID = {v: i for i, v in enumerate(VISEME_VOCAB)}
PHONEME_TO_VISEME = {p: v for v, ps in VISEME_GROUPS.items() for p in ps}
# phoneme id -> viseme id (blank -> blank); used to collapse phoneme posteriors
PHONEME_ID_TO_VISEME_ID = [0] + [VISEME_TO_ID[PHONEME_TO_VISEME[p]] for p in PHONEMES]

assert sorted(PHONEME_TO_VISEME) == sorted(PHONEMES), "every phoneme needs a viseme"

# Spelled-out letters (GRID's letter slot, acronyms) that some dictionaries lack.
LETTERS = {
    "a": "EY", "b": "B IY", "c": "S IY", "d": "D IY", "e": "IY", "f": "EH F",
    "g": "JH IY", "h": "EY CH", "i": "AY", "j": "JH EY", "k": "K EY", "l": "EH L",
    "m": "EH M", "n": "EH N", "o": "OW", "p": "P IY", "q": "K Y UW", "r": "AA R",
    "s": "EH S", "t": "T IY", "u": "Y UW", "v": "V IY", "w": "D AH B AH L Y UW",
    "x": "EH K S", "y": "W AY", "z": "Z IY",
}


@lru_cache(maxsize=1)
def _cmudict():
    """Load CMUdict as {word: [phonemes, ...]} using the first pronunciation."""
    try:
        import cmudict  # pip install cmudict (bundles the dictionary)
        raw = cmudict.dict()
    except ImportError:
        import nltk
        try:
            raw = nltk.corpus.cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            raw = nltk.corpus.cmudict.dict()
    return {w: [re.sub(r"\d", "", p) for p in prons[0]] for w, prons in raw.items()}


def normalize_text(text):
    """Lower-case, keep letters, digits, apostrophes and spaces."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def word_to_phonemes(word):
    """Phonemes for one normalised word. Unknown words fall back to spelling."""
    d = _cmudict()
    if word in d:
        return list(d[word])
    if word.isdigit():
        return [p for ch in word for p in d[_DIGITS[int(ch)]]]
    if len(word) == 1 and word in LETTERS:
        return LETTERS[word].split()
    stripped = word.strip("'")
    if stripped and stripped != word:
        return word_to_phonemes(stripped)
    # Last resort: spell the word letter by letter (rare words, names).
    return [p for ch in word if ch in LETTERS for p in LETTERS[ch].split()]


def text_to_phonemes(text, word_boundaries=False):
    """Sentence -> flat phoneme list (or list of per-word lists)."""
    words = normalize_text(text).split()
    per_word = [word_to_phonemes(w) for w in words]
    return per_word if word_boundaries else [p for ws in per_word for p in ws]


def phonemes_to_visemes(phonemes):
    return [PHONEME_TO_VISEME[p] for p in phonemes]


def text_to_visemes(text):
    return phonemes_to_visemes(text_to_phonemes(text))


def encode_phonemes(phonemes):
    return [PHONEME_TO_ID[p] for p in phonemes]


def encode_visemes(visemes):
    return [VISEME_TO_ID[v] for v in visemes]


def decode_ids(ids, vocab):
    return [vocab[i] for i in ids if i != 0]
