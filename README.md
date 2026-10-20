# See, Propose, Verify — Visually Grounded LLM Decoding for Lip Reading

> 🚧 **Work in progress** — UROP research project. Results will be added as experiments complete.
> Full method and plan: [`APPROACH.md`](APPROACH.md)

This project improves **VALLR** (Thomas, Fish & Bowden, ICCV 2025), a two-stage lip reader that predicts phonemes from video and lets a large language model (LLM) turn them into sentences. VALLR is *open-loop*: the LLM's sentence is never checked against the video, which leads to homophone errors and possible hallucination.

We **close the loop**: the LLM proposes several sentences, each is **verified against the lip movements**, and words the video does not support are **repaired**.

## Key ideas

| # | Contribution | What it does |
|---|---|---|
| **C1** | **Visual verification** | Re-scores LLM sentence candidates by the CTC likelihood of their phoneme/viseme sequence given the video, combined with the LLM score. Training-free. |
| **C2** | **Viseme-tolerant scoring** | Verification uses visemes, so candidates differing only in look-alike sounds (p/b/m, s/z) are left to the LLM, while candidates the lips contradict are rejected. |
| **C3** | **Localised repair** | Forced alignment finds words with weak visual support; the LLM rewrites only those words. |
| C4 *(extension)* | **Verified self-adaptation** | High-scoring outputs become pseudo-labels to adapt the visual encoder to a new speaker without labels. |

## Pipeline

```
Video → visual encoder → phoneme + viseme CTC
      → LLM proposes N sentences
      → VERIFY each against the video (viseme-level CTC score + LLM score)
      → REPAIR unsupported words → final sentence
```

## Datasets

| Dataset | Use | Access |
|---|---|---|
| [GRID](https://zenodo.org/record/3625687) | Visual model development, speaker adaptation | Free (CC-BY 4.0) |
| TCD-TIMIT | Full sentence-level pipeline | Free research registration |
| [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) | Comparison with VALLR | BBC data-sharing agreement |
| WikiText-103 + CMU Pronouncing Dictionary | LLM phoneme→sentence training text | Free |

Datasets are **not** included in this repository — download them from the official sources under their licences (`download_grid.sh` fetches GRID).

## Metrics

WER / CER, phoneme and viseme error rate, **unsupported-word rate** (hallucination proxy), homophone-subset accuracy, per-speaker WER, latency.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install torch torchvision opencv-python mediapipe numpy einops timm nltk tqdm tensorboard pyyaml transformers peft bitsandbytes jiwer
```

Python 3.11 recommended. LLM fine-tuning uses QLoRA (4-bit) and fits a 16 GB GPU (Kaggle / Colab).

## Roadmap

- [x] Literature review, novelty check
- [x] GRID corpus download
- [ ] Preprocessing (mouth crops, phoneme + viseme labels, WikiText phoneme corpus)
- [ ] Stage 1 — visual encoder with phoneme and viseme CTC heads
- [ ] Stage 2 — LoRA phoneme→sentence LLM (VALLR-style baseline)
- [ ] C1 + C2 — visual verification with viseme-tolerant scoring
- [ ] C3 — localised repair + unsupported-word metric
- [ ] C4 — verified speaker self-adaptation (extension)
- [ ] TCD-TIMIT / LRS2 experiments
- [ ] Paper write-up

## Results

*To be added.*

## References

Full list in [`references.bib`](references.bib). Key works:

- M. Thomas, E. Fish, R. Bowden, "VALLR: Visual ASR Language Model for Lip Reading," *ICCV*, 2025.
- C. Sun et al., "VALLR-Pin: Uncertainty-Factorized Visual Speech Recognition for Mandarin with Pinyin Guidance," arXiv:2512.20032, 2025.
- M. K. K. Teng et al., "Curriculum-Based Noise Adaptation for Phoneme-to-Text Reconstruction in Visual Speech Recognition," arXiv:2609.20839, 2026.
- A. Graves et al., "Connectionist Temporal Classification," *ICML*, 2006.

## Acknowledgements

Developed as part of an Undergraduate Research Opportunities Programme (UROP) project.
