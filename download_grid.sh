#!/usr/bin/env bash
# Downloads the GRID audio-visual corpus (normal-quality video + word alignments)
# from the University of Sheffield into ./GRID/sN/.
# Zips are staged outside OneDrive and deleted after extraction.
# Safe to re-run: finished speakers are skipped, partial downloads resume.

BASE="https://spandh.dcs.shef.ac.uk/gridcorpus"
DEST="/c/Users/Durga/OneDrive/Desktop/UROP PROJECT/GRID"
TMP="/c/Users/Durga/grid_tmp"
mkdir -p "$DEST" "$TMP"

for i in $(seq 1 34); do
  [ "$i" -eq 21 ] && continue   # speaker 21 has no video
  out="$DEST/s$i"
  if [ -f "$out/.done" ]; then echo "s$i already done, skipping"; continue; fi
  mkdir -p "$out"
  echo "=== s$i: downloading $(date +%T) ==="
  curl -fL --retry 5 --retry-delay 10 -C - -o "$TMP/s$i.zip" "$BASE/s$i/video/s$i.mpg_vcd.zip" || { echo "s$i video FAILED"; continue; }
  curl -fL --retry 5 --retry-delay 10 -o "$TMP/s$i.tar" "$BASE/s$i/align/s$i.tar" || { echo "s$i align FAILED"; continue; }
  echo "=== s$i: extracting ==="
  unzip -q -o "$TMP/s$i.zip" -d "$out" && tar -xf "$TMP/s$i.tar" -C "$out" || { echo "s$i extract FAILED"; continue; }
  rm -f "$TMP/s$i.zip" "$TMP/s$i.tar"
  touch "$out/.done"
  echo "=== s$i: done ($(find "$out" -name '*.mpg' | wc -l) videos, $(find "$out" -name '*.align' | wc -l) alignments) ==="
done
echo "ALL FINISHED $(date +%T)"
