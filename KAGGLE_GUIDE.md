# Kaggle Guide — running the lip-reading project from scratch

Your laptop has no GPU, so all heavy work runs on Kaggle (free). The code lives on GitHub; Kaggle notebooks download it and run it.

```
Laptop (code) ──push──► GitHub ──clone──► Kaggle notebooks ──► results
```

Three notebooks, run in order:

| # | Notebook | Hardware | What it does | Time |
|---|---|---|---|---|
| 01 | `01_preprocess_grid` (two copies: shard 0, shard 1) | CPU | Downloads GRID, makes mouth crops | < 12 h each |
| 02 | `02_train_visual` | GPU T4 | Trains the video → phoneme/viseme model | several hours |
| 03 | `03_llm_and_pvr` | GPU T4 | Trains the LLM, runs Propose-Verify-Repair, final results | a few hours |

---

## Part A — One-time setup (≈ 20 minutes)

### A1. Create a Kaggle account
1. Go to **https://www.kaggle.com** → **Register** (Google sign-in is fine).

### A2. Verify your phone number (required for GPU and internet)
1. Click your **profile picture** (top right) → **Settings**.
2. Find **Phone verification** → enter your number → enter the SMS code.
3. Without this, notebooks cannot use a GPU or download anything.

### A3. Know your GPU quota
- Kaggle gives roughly **30 GPU hours per week** (resets weekly) — see **Settings** / the notebook sidebar for your exact remaining quota.
- One session stops after **12 hours**. Our training code saves progress every epoch, so you can continue in a new session.
- **Always stop a GPU session when it is not running anything** (Run → Stop session / power button) — idle time still uses quota.

### A4. Hugging Face access to Llama (needed for notebook 03)
1. Create an account at **https://huggingface.co**.
2. Open **https://huggingface.co/meta-llama/Llama-3.2-1B** → read and **accept the licence** (approval usually takes minutes to hours).
3. Go to **Settings → Access Tokens → New token** → type **Read** → copy the token (starts with `hf_`).
4. Keep it private — never paste it into a notebook cell or a chat.

> No Llama access? Notebook 03 can use the open model `Qwen/Qwen2.5-1.5B` instead (change one line). Llama keeps the comparison with VALLR closest.

### A5. GitHub token (only if your repository is private)
1. GitHub → profile picture → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
2. Repository access: **only** the lip-reading repository; permission **Contents: Read-only**.
3. Copy the token (starts with `github_pat_`).

If the repository is public, skip this.

---

## Part B — Upload the notebooks to Kaggle

The notebook files are in the project folder: `UROP PROJECT\notebooks\`.

For **each** of the three `.ipynb` files:
1. Kaggle → **Create** (left menu, `+`) → **New Notebook**.
2. In the notebook: **File → Import Notebook** → choose the `.ipynb` file from `UROP PROJECT\notebooks\`.
3. Rename it (click the title at the top), e.g. `grid-preprocess-shard0`.

For notebook 01 make **two** copies: `grid-preprocess-shard0` and `grid-preprocess-shard1`.

### Add your secrets (in every notebook that needs them)
In the notebook editor: **Add-ons → Secrets → Add a new secret**
| Label | Value | Needed in |
|---|---|---|
| `GITHUB_TOKEN` | your GitHub token (only if repo is private) | 01, 02, 03 |
| `HF_TOKEN` | your Hugging Face token | 03 |

Then tick the checkbox next to each secret so the notebook may use it.

---

## Part C — Notebook 01: preprocess GRID (CPU, run twice)

### Settings (right-hand panel → **Session options**)
- **Accelerator: None** (CPU only — do not waste GPU hours here)
- **Internet: On**

### Run shard 0
1. Open `grid-preprocess-shard0`. The first code cell must read:
   ```python
   SHARD, NUM_SHARDS = 0, 2
   ```
2. Click **Save Version** (top right) → **Save & Run All (Commit)** → **Save**.
3. Kaggle runs it in the background (you can close the browser). Check progress in **View Active Events / the version's log**.
4. When it finishes, the output folder `grid_rois_shard0/` (~0.6 GB) appears in the notebook's **Output** tab.

### Run shard 1
1. Open `grid-preprocess-shard1`, change the first cell to:
   ```python
   SHARD, NUM_SHARDS = 1, 2
   ```
2. **Save Version → Save & Run All (Commit)**.

Both shards can run at the same time.

### Check
In each run's log, the last cell prints the number of processed clips (≈ 16,500 per shard) and errors (should be ~0).

---

## Part D — Notebook 02: train the visual model (GPU)

### Settings
- **Accelerator: GPU T4** (one GPU is enough; "T4 ×2" also works)
- **Internet: On**

### Add the preprocessed data
1. Right panel → **Input → Add Input** → **Your Work → Notebooks**.
2. Add the outputs of `grid-preprocess-shard0` **and** `grid-preprocess-shard1`.
3. They appear under `/kaggle/input/...`; the notebook finds them automatically.

### Run
1. **Save Version → Save & Run All (Commit)**.
2. Training runs 40 epochs and logs phoneme error (PER) and viseme error (VER) per epoch.
3. Results are in the **Output** tab: `runs/visual_grid/best.pt`, `last.pt`, `log.jsonl`.

### If it stops at 12 hours
1. Open notebook 02 → **Add Input** → add **this notebook's own previous output**.
2. In the third code cell set:
   ```python
   RESUME_FROM = '/kaggle/input/<name-of-previous-output>/runs/visual_grid'
   ```
   (use the file browser in the right panel to copy the exact path)
3. **Save & Run All** again — it continues from the last epoch.

---

## Part E — Notebook 03: LLM + Propose-Verify-Repair (GPU)

### Settings
- **Accelerator: GPU T4**, **Internet: On**
- Secret **HF_TOKEN** attached (Part B)

### Add inputs
- Outputs of `grid-preprocess-shard0` and `grid-preprocess-shard1`
- Output of notebook 02 (contains `best.pt`)

### Run
**Save Version → Save & Run All (Commit)**. It will:
1. Build the phoneme → sentence training text (GRID training sentences + 50,000 WikiText sentences).
2. Fine-tune Llama 3.2-1B with LoRA (4-bit) — 3,000 steps.
3. Run all five systems on 300 validation clips and choose the best LLM/visual weight λ.
4. Run all five systems on the **unseen-speaker test set** (speakers 1, 2, 20, 22).

### Results
In the **Output** tab:
- `runs/pvr_test/metrics.json` — WER, CER, PER and unsupported-word rate for each system
- `runs/pvr_test/outputs.jsonl` — every clip: reference, phonemes, each system's sentence, candidates and scores

Download these and give them to me — they become the results tables of the paper.

---

## Common problems

| Problem | Fix |
|---|---|
| "Internet is disabled" / downloads fail | Phone not verified (A2), or **Internet: On** not set |
| No GPU option | Phone not verified, or weekly GPU quota used up |
| `git clone` asks for a password / fails | Repo is private → add the `GITHUB_TOKEN` secret (A5, Part B) |
| `401` / `gated repo` for Llama | Licence not yet approved, or `HF_TOKEN` secret not attached |
| `CUDA out of memory` | In notebook 02, add `batch_size=8` to the `--set` part of the training cell |
| `No such file` for `/kaggle/input/...` | The input was not added, or the path differs — check the right-panel file browser |
| Session ended after 12 h | Resume as in Part D |
| Output missing after a run | Only **Save & Run All (Commit)** runs keep their output; interactive sessions do not |

---

## Order checklist

- [ ] A1–A5 one-time setup
- [ ] Code pushed to GitHub (done from the laptop)
- [ ] Notebooks imported (Part B), secrets added
- [ ] 01 shard 0 ✔  01 shard 1 ✔
- [ ] 02 visual model ✔ (`best.pt` exists)
- [ ] 03 LLM + PVR ✔ (`metrics.json` exists)
- [ ] Results downloaded and shared for the paper
