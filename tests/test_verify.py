"""Checks for the Verify step on synthetic CTC posteriors (run: python -m pytest tests)."""
import torch

from src.pvr.verify import ctc_logprob, forced_align, rerank, sentence_ids, word_support
from src.text.phonemes import VISEME_VOCAB


def fake_posteriors(text, level="viseme", frames_per_token=3, vocab=len(VISEME_VOCAB)):
    """Posteriors that clearly 'show' `text`: each token for a few frames, blanks between."""
    ids, _ = sentence_ids(text, level)
    rows = []
    for i in ids:
        rows += [i] * frames_per_token + [0]
    logits = torch.full((len(rows), vocab), -6.0)
    for t, k in enumerate(rows):
        logits[t, k] = 4.0
    return torch.log_softmax(logits, -1)


def test_true_sentence_scores_highest():
    lp = fake_posteriors("bin blue at f two now")
    good = ctc_logprob(lp, sentence_ids("bin blue at f two now", "viseme")[0])
    bad = ctc_logprob(lp, sentence_ids("set green with q nine again", "viseme")[0])
    assert good > bad + 10


def test_viseme_level_cannot_separate_homophenes():
    lp = fake_posteriors("pat")
    assert abs(ctc_logprob(lp, sentence_ids("pat", "viseme")[0])
               - ctc_logprob(lp, sentence_ids("bat", "viseme")[0])) < 1e-6


def test_rerank_lets_language_model_break_ties():
    lp = fake_posteriors("bin blue at p two now")
    ranked = rerank([("bin blue at b two now", -5.0), ("bin blue at p two now", -1.0),
                     ("set green with q nine again", 0.0)], lp, "viseme", lam=0.5)
    assert ranked[0]["text"] == "bin blue at p two now"      # same visemes as "b", better LLM score
    assert ranked[-1]["text"] == "set green with q nine again"  # contradicted by the video


def test_forced_align_and_word_support():
    text = "bin blue at f two now"
    lp = fake_posteriors(text)
    ids, _ = sentence_ids(text, "viseme")
    frames = forced_align(lp, ids)
    assert frames is not None and all(len(f) == 3 for f in frames)
    support = word_support(lp, text)
    assert all(s > -0.5 for _, s in support)                  # every word is supported
    wrong = word_support(lp, "bin blue at f nine now")
    assert wrong[4][1] < min(s for i, (_, s) in enumerate(wrong) if i != 4)
