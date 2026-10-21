"""Verify: score sentence candidates against the video, and locate unsupported words.

visual score  = log P_CTC(sequence of the sentence | video), at phoneme or viseme level.
At viseme level, sentences that differ only in look-alike sounds (p/b/m, s/z, ...)
receive the same visual score, so the language model decides between them,
while sentences the lips contradict are penalised.
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.text.phonemes import (encode_phonemes, encode_visemes, phonemes_to_visemes,
                               text_to_phonemes)

NEG = -1e9


def sentence_ids(text, level):
    """Token ids of a sentence at 'phoneme' or 'viseme' level, plus per-word token counts."""
    words = text_to_phonemes(text, word_boundaries=True)
    flat = [p for w in words for p in w]
    ids = encode_phonemes(flat) if level == "phoneme" else encode_visemes(phonemes_to_visemes(flat))
    return ids, [len(w) for w in words]


def ctc_logprob(logp, ids):
    """log P(ids | video) under CTC. logp: (T, V) log-probabilities for one clip."""
    if not ids:
        return NEG
    loss = F.ctc_loss(logp.unsqueeze(1).float(), torch.tensor([ids]), torch.tensor([logp.shape[0]]),
                      torch.tensor([len(ids)]), blank=0, reduction="sum", zero_infinity=False)
    v = -loss.item()
    return v if np.isfinite(v) else NEG


def rerank(cands, logp, level, lam):
    """cands: [(text, llm_score)] -> list of dicts sorted by combined score (best first).

    combined = lam * llm_score + (1 - lam) * visual_score
    """
    scored = []
    for text, llm in cands:
        ids, _ = sentence_ids(text, level)
        vis = ctc_logprob(logp, ids)
        scored.append({"text": text, "llm": llm, "visual": vis, "score": lam * llm + (1 - lam) * vis})
    return sorted(scored, key=lambda d: d["score"], reverse=True)


def forced_align(logp, ids):
    """CTC Viterbi alignment. Returns, for each target token, the list of frames it occupies
    (None if the sequence cannot be aligned to this many frames)."""
    lp = logp.detach().cpu().numpy()
    T, L = lp.shape[0], len(ids)
    ext = [0]
    for i in ids:
        ext += [i, 0]
    S = len(ext)
    dp = np.full((T, S), -np.inf)
    back = np.zeros((T, S), dtype=np.int64)
    dp[0, 0] = lp[0, ext[0]]
    if S > 1:
        dp[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(dp[t - 1, s], s)]
            if s >= 1:
                cands.append((dp[t - 1, s - 1], s - 1))
            if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2]:
                cands.append((dp[t - 1, s - 2], s - 2))
            best, arg = max(cands)
            dp[t, s], back[t, s] = best + lp[t, ext[s]], arg
    end = S - 1 if S == 1 or dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    if not np.isfinite(dp[T - 1, end]):
        return None
    frames = [[] for _ in range(L)]
    s = end
    for t in range(T - 1, -1, -1):
        if ext[s] != 0:
            frames[(s - 1) // 2].append(t)
        s = back[t, s] if t > 0 else s
    return [sorted(f) for f in frames]


def word_support(logp, text, level="viseme"):
    """Per-word visual support: mean over the word's aligned frames of
    log p(token) - max_v log p(v)   (0 = the video's top choice; very negative = unsupported)."""
    ids, counts = sentence_ids(text, level)
    frames = forced_align(logp, ids)
    words = text.split()
    if frames is None:
        return [(w, -np.inf) for w in words]
    lp = logp.detach().cpu().numpy()
    best = lp.max(1)
    out, k = [], 0
    for w, c in zip(words, counts):
        vals = [lp[t, ids[j]] - best[t] for j in range(k, k + c) for t in frames[j]]
        out.append((w, float(np.mean(vals)) if vals else -np.inf))
        k += c
    return out


def unsupported_words(logp, text, tau, level="viseme"):
    return [i for i, (_, s) in enumerate(word_support(logp, text, level)) if s < tau]
