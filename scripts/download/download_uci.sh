#!/bin/bash
set -euo pipefail

ROOT="${HETEROWEAR_ROOT:-.}/data/raw"

mkdir -p \
    "$ROOT/ppg_dalia" \
    "$ROOT/pamap2" \
    "$ROOT/wisdm" \
    "$ROOT/hhar" \
    "$ROOT/harth" \
    "$ROOT/uci_har"

download_dataset() {
    NAME="$1"
    URL="$2"
    OUTPUT="$3"

    echo
    echo "============================================================"
    echo "Dataset: $NAME"
    echo "Started: $(date)"
    echo "Output:  $OUTPUT"
    echo "============================================================"

    wget \
        -c \
        --show-progress \
        --tries=10 \
        --timeout=60 \
        "$URL" \
        -O "$OUTPUT"

    echo "$NAME completed: $(date)"
    ls -lh "$OUTPUT"
}


download_dataset \
    "PPG-DaLiA" \
    "https://archive.ics.uci.edu/static/public/495/ppg%2Bdalia.zip" \
    "$ROOT/ppg_dalia/ppg_dalia.zip"


download_dataset \
    "PAMAP2" \
    "https://archive.ics.uci.edu/static/public/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring.zip" \
    "$ROOT/pamap2/pamap2.zip"


download_dataset \
    "WISDM" \
    "https://archive.ics.uci.edu/static/public/507/wisdm%2Bsmartphone%2Band%2Bsmartwatch%2Bactivity%2Band%2Bbiometrics%2Bdataset.zip" \
    "$ROOT/wisdm/wisdm.zip"


download_dataset \
    "HHAR" \
    "https://archive.ics.uci.edu/static/public/344/heterogeneity%2Bactivity%2Brecognition.zip" \
    "$ROOT/hhar/hhar.zip"


download_dataset \
    "HARTH" \
    "https://archive.ics.uci.edu/static/public/779/harth.zip" \
    "$ROOT/harth/harth.zip"


download_dataset \
    "UCI-HAR-Raw" \
    "https://archive.ics.uci.edu/static/public/341/smartphone%2Bbased%2Brecognition%2Bof%2Bhuman%2Bactivities%2Band%2Bpostural%2Btransitions.zip" \
    "$ROOT/uci_har/uci_har_raw.zip"


echo
echo "============================================================"
echo "ALL DOWNLOADS COMPLETE"
echo "$(date)"
echo "============================================================"

du -sh "$ROOT"/*
