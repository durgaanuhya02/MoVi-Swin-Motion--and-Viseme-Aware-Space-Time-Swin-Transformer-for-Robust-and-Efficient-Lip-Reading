"""Generates the Kaggle notebooks in this folder (run: python notebooks/make_notebooks.py).

01_preprocess_grid  CPU   download GRID, extract mouth ROIs for one shard (run once per shard)
02_train_visual     GPU   train the phoneme/viseme CTC visual model (re-run to resume)
03_llm_and_pvr      GPU   build LLM corpus, LoRA fine-tune, run Propose-Verify-Repair evaluation
"""
import json
from pathlib import Path

REPO = "https://github.com/durgaanuhya02/MoVi-Swin-Motion--and-Viseme-Aware-Space-Time-Swin-Transformer-for-Robust-and-Efficient-Lip-Reading.git"

SETUP = f"""# --- setup: clone the project code (private repo: add a Kaggle secret GITHUB_TOKEN) ---
import os, subprocess
url = "{REPO}"
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
    url = url.replace("https://", f"https://{{tok}}@")
except Exception:
    pass
if not os.path.exists("/kaggle/working/repo"):
    subprocess.run(["git", "clone", "--depth", "1", url, "/kaggle/working/repo"], check=True)
os.chdir("/kaggle/working/repo")
!git log --oneline -1"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def notebook(cells, gpu):
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none",
                                    "isInternetEnabled": True}}}


preprocess = notebook([
    md("# 01 · Preprocess GRID (CPU)\n"
       "Downloads GRID from the official Sheffield site into `/tmp`, extracts 96×96 mouth ROIs + lip "
       "landmarks for **one shard**, and writes them to `/kaggle/working/grid_rois_shard{SHARD}`.\n\n"
       "Run this notebook once per shard (`SHARD = 0`, then `1`) with **Save Version → Save & Run All**. "
       "Then create a Kaggle Dataset from each version's output (e.g. `grid-rois-shard0`, `grid-rois-shard1`).\n\n"
       "Settings: Accelerator **None**, Internet **On**."),
    code("SHARD, NUM_SHARDS = 0, 2      # change to 1 for the second run"),
    code(SETUP),
    code("!pip install -q mediapipe cmudict"),
    code("!df -h /tmp /kaggle/working\n!nproc"),
    code("!GRID_DEST=/tmp/GRID GRID_TMP=/tmp/grid_zips bash download_grid.sh 2>&1 | grep -E 'done|FAILED|FINISHED'"),
    code("!python -m src.preprocess.build_grid_manifest --grid-root /tmp/GRID --out /kaggle/working/grid_manifest.jsonl"),
    code("OUT = f'/kaggle/working/grid_rois_shard{SHARD}'\n"
         "!python -m src.preprocess.extract_rois --manifest /kaggle/working/grid_manifest.jsonl --out {OUT} "
         "--workers 4 --detect-every 2 --shard {SHARD}/{NUM_SHARDS} 2>&1 | grep -E 'clips|eta' | tail -20"),
    code("!wc -l {OUT}/index.jsonl {OUT}/errors.jsonl; du -sh {OUT}\n"
         "!rm -rf /kaggle/working/repo   # keep only the data in the saved output"),
], gpu=False)

train_visual = notebook([
    md("# 02 · Train the visual model (GPU)\n"
       "Add the datasets `grid-rois-shard0` and `grid-rois-shard1` (**Add Data**). Accelerator **GPU T4**.\n\n"
       "Training saves `last.pt` every epoch. If the 12-hour session ends, add this notebook's previous "
       "output as data, set `RESUME_FROM` to its `runs/visual_grid` folder and run again."),
    code(SETUP),
    code("!pip install -q cmudict"),
    code("import glob, json\nINDEX = sorted(glob.glob('/kaggle/input/**/grid_rois_shard*/index.jsonl', recursive=True))\n"
         "INDEX_JSON = json.dumps(INDEX)\nprint(INDEX)\n"
         "RESUME_FROM = None   # e.g. '/kaggle/input/<previous-output>/runs/visual_grid'"),
    code("import shutil, os\nOUT = '/kaggle/working/runs/visual_grid'\nos.makedirs(OUT, exist_ok=True)\n"
         "if RESUME_FROM:\n    for f in ['last.pt', 'best.pt', 'log.jsonl']:\n"
         "        if os.path.exists(f'{RESUME_FROM}/{f}'): shutil.copy(f'{RESUME_FROM}/{f}', OUT)"),
    code("!python -m src.train_visual --config configs/visual_grid.yaml "
         "--set 'index={INDEX_JSON}' out_dir={OUT}"),
    code("!tail -5 {OUT}/log.jsonl"),
], gpu=True)

llm_pvr = notebook([
    md("# 03 · LLM fine-tuning + Propose-Verify-Repair (GPU)\n"
       "Add data: both ROI shards and the output of notebook 02 (for `best.pt`). Accelerator **GPU T4**.\n\n"
       "Llama 3.2 is gated: accept its licence on huggingface.co and add a Kaggle secret **HF_TOKEN**. "
       "Without it, set `LLM` to an open model such as `Qwen/Qwen2.5-1.5B`."),
    code(SETUP),
    code("!pip install -q cmudict peft bitsandbytes"),
    code("import glob, os\nINDEX = sorted(glob.glob('/kaggle/input/**/grid_rois_shard*/index.jsonl', recursive=True))\n"
         "VISUAL = glob.glob('/kaggle/input/**/visual_grid/best.pt', recursive=True)[0]\n"
         "LLM = 'meta-llama/Llama-3.2-1B'\n"
         "try:\n    from kaggle_secrets import UserSecretsClient\n"
         "    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')\nexcept Exception as e:\n"
         "    print('no HF_TOKEN secret:', e)\nINDEX_ARGS = ' '.join(INDEX)\nprint(INDEX, VISUAL)"),
    md("## Stage 2 · phoneme→sentence corpus and LoRA fine-tuning"),
    code("!cat {INDEX_ARGS} > /kaggle/working/grid_index_all.jsonl\n"
         "!python -m src.llm.build_corpus --manifest /kaggle/working/grid_index_all.jsonl --split train "
         "--wikitext 50000 --out /kaggle/working/llm_corpus.jsonl"),
    code("!python -m src.llm.train_p2t --config configs/llm_p2t.yaml --set model_name={LLM} max_steps=3000"),
    md("## Stage 3 · tune λ on validation, then evaluate on the unseen-speaker test set"),
    code("ADAPTER = '/kaggle/working/runs/llm_p2t/adapter'\n"
         "!python -m src.pvr.run --visual {VISUAL} --index {INDEX_ARGS} --split val --llm {LLM} "
         "--adapter {ADAPTER} --out /kaggle/working/runs/pvr_val --n 5 --limit 300 --sweep | tail -40"),
    code("import json\nsweep = json.load(open('/kaggle/working/runs/pvr_val/metrics.json'))['lambda_sweep']\n"
         "LAM = float(min(sweep, key=sweep.get)); print('best lambda', LAM, sweep)"),
    code("!python -m src.pvr.run --visual {VISUAL} --index {INDEX_ARGS} --split test --llm {LLM} "
         "--adapter {ADAPTER} --out /kaggle/working/runs/pvr_test --n 5 --lam {LAM} --sweep | tail -40"),
], gpu=True)

here = Path(__file__).parent
for name, nb in [("01_preprocess_grid", preprocess), ("02_train_visual", train_visual), ("03_llm_and_pvr", llm_pvr)]:
    (here / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    print("wrote", name)
