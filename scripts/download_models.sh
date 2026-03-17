#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$MODEL_DIR"

FILES="config.json tokenizer.json model.safetensors"

download_model() {
    local repo=$1
    local name=$2
    local dir="$MODEL_DIR/$name"

    if [ -f "$dir/model.safetensors" ]; then
        echo "[$name] already exists, skipping"
        return
    fi

    echo "[$name] downloading to $dir"
    mkdir -p "$dir"
    for f in $FILES; do
        echo "  $f"
        curl -sL "https://huggingface.co/$repo/resolve/main/$f" -o "$dir/$f"
    done
    echo "[$name] done"
}

download_model "HuggingFaceTB/SmolLM-135M"  "SmolLM-135M"
download_model "Qwen/Qwen2.5-0.5B"         "Qwen2.5-0.5B"
download_model "Qwen/Qwen2.5-0.5B-Instruct" "Qwen2.5-0.5B-Instruct"

echo ""
echo "All models downloaded to $MODEL_DIR"
