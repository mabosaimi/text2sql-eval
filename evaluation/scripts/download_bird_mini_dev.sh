#!/bin/bash
# This script downloads Bird mini-dev (~3.3GB unzipped) from the official download link.
# Source: https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG
# GitHub: https://github.com/bird-bench/mini_dev
set -e

DATA_DIR="evaluation/tasks/bird_mini_dev/data"
mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/minidev/MINIDEV/mini_dev_sqlite.json" ]; then
    echo "Bird mini-dev already exists, skipping download"
    exit 0
fi

echo "Downloading Bird mini-dev dataset..."
gdown "13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG" -O "$DATA_DIR/minidev_0703.zip"

echo "Extracting..."
unzip -q "$DATA_DIR/minidev_0703.zip" -d "$DATA_DIR"
rm "$DATA_DIR/minidev_0703.zip"

echo "Bird mini-dev download complete!"
