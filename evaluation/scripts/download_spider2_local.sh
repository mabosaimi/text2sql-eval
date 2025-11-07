#!/bin/bash
# This script downloads Spider2 local_db (~1.6GB unzipped) from the official download link.
# Source: https://drive.usercontent.google.com/download?id=1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG
# GitHub: https://github.com/xlang-ai/Spider2/tree/main/spider2-lite
set -e

DATA_DIR="evaluation/tasks/spider2_local/data"
mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/local-map.jsonl" ]; then
    echo "Spider2 local already exists, skipping download"
    exit 0
fi

echo "Downloading Spider2 local dataset..."
gdown "1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG" -O "$DATA_DIR/spider2_local.zip"

echo "Extracting..."
unzip -q "$DATA_DIR/spider2_local.zip" -d "$DATA_DIR"
rm "$DATA_DIR/spider2_local.zip"

echo "Spider2 local download complete!"
