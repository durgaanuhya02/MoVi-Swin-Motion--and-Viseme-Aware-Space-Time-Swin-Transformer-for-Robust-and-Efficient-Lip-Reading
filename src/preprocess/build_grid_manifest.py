"""Build a manifest (JSON lines) for the GRID corpus.

Every dataset (GRID now, VoxMM later) is converted to the same manifest format,
so the rest of the pipeline never needs dataset-specific code:

    {"id": "s1/bbaf2n", "video": "<abs path>", "start": 0, "end": null,
     "text": "bin blue at f two now", "speaker": "s1", "split": "train"}

`start`/`end` are frame indices (end exclusive, null = until the last frame).

Split: unseen-speaker protocol of LipNet (Assael et al., 2016) — speakers
1, 2, 20 and 22 are held out for test; 5% of the remaining clips form a
validation set (seen speakers).

Usage:
    python -m src.preprocess.build_grid_manifest --grid-root GRID --out data/grid_manifest.jsonl
"""
import argparse
import json
import random
from pathlib import Path

TEST_SPEAKERS = {"s1", "s2", "s20", "s22"}
SILENCE = {"sil", "sp"}


def read_align(path):
    """Return the spoken words of a GRID .align file (silences removed)."""
    words = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] not in SILENCE:
            words.append(parts[2])
    return words


def build(grid_root, val_fraction=0.05, seed=0):
    grid_root = Path(grid_root).resolve()
    entries = []
    for spk_dir in sorted(grid_root.glob("s*"), key=lambda p: int(p.name[1:])):
        spk = spk_dir.name
        aligns = {p.stem: p for p in spk_dir.rglob("*.align")}
        for video in sorted(spk_dir.rglob("*.mpg")):
            align = aligns.get(video.stem)
            if align is None:
                continue
            words = read_align(align)
            if not words:
                continue
            entries.append({
                "id": f"{spk}/{video.stem}",
                "video": str(video),
                "start": 0,
                "end": None,
                "text": " ".join(words),
                "speaker": spk,
                "split": "test" if spk in TEST_SPEAKERS else "train",
            })

    rng = random.Random(seed)
    train = [e for e in entries if e["split"] == "train"]
    for e in rng.sample(train, int(len(train) * val_fraction)):
        e["split"] = "val"
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    args = ap.parse_args()

    entries = build(args.grid_root, args.val_fraction)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    counts = {s: sum(e["split"] == s for e in entries) for s in ("train", "val", "test")}
    print(f"wrote {len(entries)} clips to {out}: {counts}")


if __name__ == "__main__":
    main()
