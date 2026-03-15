#!/usr/bin/env bash
# Download GPT-2 ONNX model files and tokenizer data to tests/fixtures/
set -euo pipefail

FIXTURES_DIR="$(cd "$(dirname "$0")/.." && pwd)/tests/fixtures"
BASE_URL="https://huggingface.co/openai-community/gpt2/resolve/main/onnx"

files=(
    "decoder_model.onnx:gpt2_decoder_model.onnx"
    "decoder_with_past_model.onnx:gpt2_decoder_with_past_model.onnx"
    "decoder_model_merged.onnx:gpt2_decoder_model_merged.onnx"
    "vocab.json:vocab.json"
    "merges.txt:merges.txt"
)

mkdir -p "$FIXTURES_DIR"

for entry in "${files[@]}"; do
    remote="${entry%%:*}"
    local="${entry##*:}"
    dest="$FIXTURES_DIR/$local"

    if [ -f "$dest" ]; then
        echo "  skip $local (already exists)"
        continue
    fi

    echo "  downloading $local ..."
    curl -fSL -o "$dest" "$BASE_URL/$remote"
done

echo "done. files in $FIXTURES_DIR:"
ls -lh "$FIXTURES_DIR"/gpt2_* "$FIXTURES_DIR"/vocab.json "$FIXTURES_DIR"/merges.txt 2>/dev/null
