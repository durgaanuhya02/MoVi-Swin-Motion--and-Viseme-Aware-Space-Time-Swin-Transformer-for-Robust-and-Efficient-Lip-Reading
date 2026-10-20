# Final Approach — Propose · Verify · Repair (PVR)

**Working title:** *See, Propose, Verify: Visually Grounded LLM Decoding for Phoneme-Centric Lip Reading*

**Base paper we improve:** M. Thomas, E. Fish, R. Bowden, *"VALLR: Visual ASR Language Model for Lip Reading,"* ICCV 2025 ([arXiv 2503.21408](https://arxiv.org/abs/2503.21408)).

---

## 1. Problem with the base paper

VALLR is **open-loop**: the video model predicts one phoneme string, the LLM rewrites it into a sentence, and the sentence is **never checked against the video**. The paper itself reports the two consequences:

1. **Homophone / homophene errors** — e.g. *your* vs *you're*, /s/ vs /z/ (Sec. 5, Conclusion).
2. **Hallucination risk** — a fluent sentence that the lips do not support (Sec. 1), which is never measured.

## 2. Novelty check (Oct 2026)

| Idea | Status | Prior work |
|---|---|---|
| N-best hypotheses as LLM input | ❌ already done | VALLR-Pin (arXiv 2512.20032), DualHyp (arXiv 2510.13281) |
| LLM trained on realistic phoneme errors | ❌ already done | PECT (arXiv 2609.20839) |
| Visemes instead of phonemes as LLM input | ❌ tested — phonemes better | "Tackling the Phoneme-to-Text Conversion Bottleneck…" (TCD-TIMIT) |
| Pseudo-label speaker adaptation | ❌ already done | Kim et al. 2022 (arXiv 2208.04498); ACM 2024 TTA |
| **Re-scoring LLM sentence candidates by their fit to the video** | ✅ **not found** | closest: DLLM-VSR reranks by *length* plausibility (arXiv 2605.28456) |
| **Localising unsupported words and repairing only those** | ✅ **not found** | — |
| **Viseme-tolerant verification score** | ✅ **not found** | — |

→ We build on the open gaps and use the "done" ideas only as **baselines**.

## 3. Method

```
Video ──► (1) SEE: visual encoder ──► phoneme CTC posteriors  +  viseme CTC posteriors (auxiliary head)
                                         │
                                         ▼  1-best phonemes (as in VALLR)
          (2) PROPOSE: LLM ──► N candidate sentences  (beam / sampling)
                                         │
                                         ▼
          (3) VERIFY: each sentence → phonemes (CMU dict) → visemes
                      visual score = log P_CTC(viseme seq | video)      ← viseme-tolerant
                      final score  = λ · LLM score + (1 − λ) · visual score
                                         │
                                         ▼
          (4) REPAIR: forced-align best sentence to the video; words whose
                      segment score < τ are marked [?] and the LLM rewrites
                      only those words (≤ 2 iterations)
                                         │
                                         ▼
                                   final sentence
```

### C1 — Visual verification (core contribution)
- Generate N candidates (N = 5–10) from the phoneme-to-text LLM.
- Convert each candidate to phonemes (CMU dict, G2P fallback for unknown words) and then visemes.
- Score with the **CTC log-likelihood of the candidate's sequence given the video** (`torch.nn.functional.ctc_loss`, one call per candidate) and combine with the LLM log-probability. λ tuned on validation.
- **Training-free** on the LLM side → cheap; works on top of any CTC visual model.

### C2 — Viseme-tolerant scoring (principled split of responsibility)
- Verify with the **viseme** head, not the phoneme head: candidates differing only inside a viseme class (p/b/m, s/z, f/v) get the **same visual score**, so the decision is left to the LLM; candidates the lips actually contradict are penalised.
- Rule: *vision judges what is visible, language decides what is not.*
- Ablation: phoneme-level vs viseme-level vs mixed verification.

### C3 — Localised repair
- CTC forced alignment gives a per-word visual support score.
- Words below threshold τ are replaced with a mask token and the LLM is asked to fill only those spans; the result is re-verified.
- Directly targets hallucinated / unsupported words instead of regenerating the whole sentence.

### C4 (extension) — Verified self-adaptation
- Use sentences with high verified score as pseudo-labels to adapt a LoRA module in the visual encoder per speaker (no labels).
- Comparison point: plain confidence-thresholded self-training (Kim et al. 2022). Our claim to test: *verification is a better pseudo-label filter than model confidence.*

## 4. Baselines

| Baseline | Purpose |
|---|---|
| VALLR-style 1-best → LLM (reproduced) | Main baseline |
| + N-best input to LLM (VALLR-Pin style) | Prior-work baseline for C1 |
| + LLM reranking by LLM score only | Shows the visual score matters |
| CTC-only phoneme decoding + lexicon | Lower bound |

## 5. Metrics

| Metric | Measures |
|---|---|
| WER / CER | Standard accuracy |
| PER, VER | Phoneme / viseme error rate of Stage 1 |
| **Unsupported-word rate** (new) | % of output words whose aligned visual support < τ — hallucination proxy |
| Homophone / homophene subset accuracy | Ambiguity handling |
| Per-speaker WER before/after adaptation | C4 |
| Latency, #LLM calls | Cost of verify + repair |
| Mean ± std over 3 seeds | Reliability |

## 6. Data

| Dataset | Role | Access |
|---|---|---|
| GRID (downloaded) | Stage-1 development (PER/VER), C4 speaker adaptation (1000 clips/speaker) | ✅ in `GRID/` |
| TCD-TIMIT | Full pipeline on real sentences | Free research registration |
| LRS2 | Head-to-head with VALLR (20.8% WER) | BBC agreement — **apply now** |
| WikiText-103 + CMU dict | LLM phoneme→sentence training text | Free |

Note: GRID's fixed grammar makes the LLM stage trivial, so sentence-level claims need TCD-TIMIT / LRS2.

## 7. Compute

- Visual encoder: Kaggle / Colab GPU.
- LLM: Llama 3.2-1B (main) and 3B (if time), **QLoRA 4-bit** → fits a 16 GB T4.
- Verify + repair: inference only.

## 8. Plan

| Weeks | Task |
|---|---|
| 1–2 | Preprocessing (GRID, TCD-TIMIT), phoneme + viseme labels, WikiText phoneme corpus |
| 3–4 | Stage 1: visual encoder + phoneme CTC + viseme CTC heads; PER/VER on GRID |
| 5 | Stage 2: LoRA phoneme→sentence LLM; reproduce VALLR-style baseline |
| 6–7 | C1 verify + C2 viseme-tolerant scoring; ablations on λ, N |
| 8 | C3 repair + unsupported-word metric |
| 9–10 | C4 adaptation on GRID speakers (if time); LRS2 if licence granted |
| 11–12 | Evaluation, paper writing |

## 9. Risks

| Risk | Mitigation |
|---|---|
| Visual score too weak to separate candidates | Fall back to phoneme-level score; tune λ; report as finding |
| Concurrent publication of verify-style rescoring | Keep C2 (viseme-tolerant) + C3 (repair) + new metric as distinct contributions; re-check arXiv monthly |
| No LRS2 licence | Report TCD-TIMIT + GRID; frame as data-efficient study |
| LLM too slow | 1B model, small N, cache candidates |

## 10. Relation to MoVi-Swin

The MoVi-Swin encoder ideas (space-time windows, landmark stream) remain an **optional Stage-1 encoder upgrade**. The core contribution is now the verification loop, which works with any CTC visual encoder.
