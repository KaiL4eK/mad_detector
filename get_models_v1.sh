#!/usr/bin/env bash

HOST="194.85.169.205:8000"

mkdir -p models; cd models

# VERSION="RF_model_YOLO_MbN"
# BASE_MODEL="MbN2_416x416_t1"

VERSION="RF_model_YOLO_Tiny"
BASE_MODEL="Tiny3_416x416_t1"

FP16_MODEL="$BASE_MODEL"_FP16
FP32_MODEL="$BASE_MODEL"_FP32

BASE_URL="http://$HOST/test_data/$VERSION"

wget "$BASE_URL/$BASE_MODEL.json" -O "model.json"

wget "$BASE_URL/$FP16_MODEL.bin" -O "model.bin"
wget "$BASE_URL/$FP16_MODEL.mapping" -O "model.mapping"
wget "$BASE_URL/$FP16_MODEL.xml" -O "model.xml"

# wget -N "$BASE_URL/$FP32_MODEL.bin"
# wget -N "$BASE_URL/$FP32_MODEL.mapping"
# wget -N "$BASE_URL/$FP32_MODEL.xml"
