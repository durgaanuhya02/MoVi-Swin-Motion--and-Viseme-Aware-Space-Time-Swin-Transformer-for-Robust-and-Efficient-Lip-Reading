"""Error-rate metrics: WER, CER, phoneme (PER) and viseme (VER) error rate."""


def edit_distance(ref, hyp):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def error_rate(refs, hyps):
    """Corpus-level error rate = total edits / total reference length."""
    edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return edits / max(total, 1)


def wer(ref_texts, hyp_texts):
    return error_rate([r.split() for r in ref_texts], [h.split() for h in hyp_texts])


def cer(ref_texts, hyp_texts):
    return error_rate(list(ref_texts), list(hyp_texts))
