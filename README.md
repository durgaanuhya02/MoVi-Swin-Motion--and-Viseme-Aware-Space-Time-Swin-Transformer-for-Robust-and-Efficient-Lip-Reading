# MoVi-Swin: Motion- and Viseme-Aware Space-Time Swin Transformer for Lip Reading



MoVi-Swin is a lightweight visual speech recognition (lip reading) model that recognises spoken words from silent video of the mouth. It builds on two recent transformer lip readers — **SwinLip** (Park et al., 2025) and **3DCvT** (Wang et al., 2022) — and targets the core difficulty of lip reading: **homophenes**, words that look the same on the lips (e.g. *b / p / m*).

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

