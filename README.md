# MoVi-Swin: Motion- and Viseme-Aware Space-Time Swin Transformer for Lip Reading

> 🚧 **Work in progress** — UROP research project. Results will be added as experiments complete.

MoVi-Swin is a lightweight visual speech recognition (lip reading) model that recognises spoken words from silent video of the mouth. It builds on two recent transformer lip readers — **SwinLip** (Park et al., 2025) and **3DCvT** (Wang et al., 2022) — and targets the core difficulty of lip reading: **homophenes**, words that look the same on the lips (e.g. *b / p / m*).

## Key ideas

| # | Contribution | What it does |
|---|---|---|
| **C1** | **Space-time shifted-window encoder + SE-Patch-Merging** | Extends SwinLip's per-frame 2D attention to 3D windows spanning neighbouring frames, and re-weights channels (3DCvT's SE idea) where downsampling loses information. |
| **C2** | **Landmark motion stream** | A spatio-temporal graph network over 40 MediaPipe lip landmarks (position + velocity), fused into the transformer at multiple stages through gated cross-attention for robustness to pose and lighting. |
| **C3** | **Viseme-supervised training** | An auxiliary CTC loss over viseme (visual phoneme) sequences inside the encoder, plus a homophene-group evaluation. |
| C4 *(optional)* | Audio-teacher distillation | Audio is used only during training; inference is visual-only. |

## Architecture

```
Mouth video (T×88×88)                    Lip landmarks (T×40×3)
      │                                          │
3D spatio-temporal stem                   ST-GCN motion encoder
      │                                          │
Space-time Swin stages + SE-Patch-Merging ◄── gated cross-attention
      │
      ├── viseme CTC head (auxiliary)
      │
1D Conv-Attention → temporal back-end → word classifier
```

## Datasets

| Dataset | Use | Access |
|---|---|---|
| [GRID](https://zenodo.org/record/3625687) | Development, ablations, homophene analysis (unseen-speaker split: speakers 1, 2, 20, 22 held out) | Free (CC-BY 4.0) |
| [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) | Main English benchmark (500 words) | BBC data-sharing agreement |
| [LRW-1000](https://vipl.ict.ac.cn/resources/databases/201810/t20181017_32714.html) | Mandarin benchmark (1000 words) | VIPL release agreement |

Datasets are **not** included in this repository — download them from the official sources under their licences.

## Project structure

```
src/
├─ preprocess/   mouth cropping, lip landmarks, word clipping, viseme labels
├─ datasets/     PyTorch datasets and augmentation
├─ models/       baseline, SwinLip, MoVi-Swin and building blocks
├─ train.py
└─ evaluate.py   accuracy, homophene-group accuracy, FLOPs, latency
configs/         one YAML file per experiment
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install torch torchvision opencv-python mediapipe numpy einops timm fvcore nltk tqdm tensorboard pyyaml
```

Python 3.11 is recommended (MediaPipe compatibility).

## Roadmap

- [x] Literature review and dataset selection
- [x] GRID corpus download
- [ ] Preprocessing pipeline (mouth crops, landmarks, word clips, visemes)
- [ ] Baseline (3D conv + ResNet-18 + BiGRU)
- [ ] SwinLip reproduction
- [ ] C1 — space-time windows + SE-Patch-Merging
- [ ] C2 — landmark motion stream
- [ ] C3 — viseme CTC supervision
- [ ] LRW / LRW-1000 experiments
- [ ] Paper write-up

## Results

*To be added.*

## References

Key works this project builds on (full list in [`references.bib`](references.bib)):

- Y.-H. Park, R.-H. Park, H.-M. Park, "SwinLip: An Efficient Visual Speech Encoder for Lip Reading Using Swin Transformer," *Neurocomputing*, 2025.
- H. Wang, G. Pu, T. Chen, "A Lip Reading Method Based on 3D Convolutional Vision Transformer," *IEEE Access*, 2022.
- Y. M. Assael et al., "LipNet: End-to-End Sentence-level Lipreading," arXiv:1611.01599, 2016.
- M. Cooke et al., "An Audio-Visual Corpus for Speech Perception and Automatic Speech Recognition," *JASA*, 2006.

## Acknowledgements

Developed as part of an Undergraduate Research Opportunities Programme (UROP) project.
