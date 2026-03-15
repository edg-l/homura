#!/bin/bash
mkdir -p tests/fixtures
curl -L -o tests/fixtures/mnist-12.onnx \
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
echo "Downloaded mnist-12.onnx"
