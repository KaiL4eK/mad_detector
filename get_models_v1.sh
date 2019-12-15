#!/usr/bin/env bash

HOST="194.85.169.205:8000"

mkdir -p models; cd models

VERSION="RF_model_YOLO"

BASE_MODEL="MbN2_416x416_t1"

FP16_MODEL="$BASE_MODEL"_FP16
FP32_MODEL="$BASE_MODEL"_FP32

BASE_URL="http://$HOST/test_data/$VERSION"

wget -N "$BASE_URL/$BASE_MODEL.json"

wget -N "$BASE_URL/$FP16_MODEL.bin"
wget -N "$BASE_URL/$FP16_MODEL.mapping"
wget -N "$BASE_URL/$FP16_MODEL.xml"

# wget -N "$BASE_URL/$FP32_MODEL.bin"
# wget -N "$BASE_URL/$FP32_MODEL.mapping"
# wget -N "$BASE_URL/$FP32_MODEL.xml"
