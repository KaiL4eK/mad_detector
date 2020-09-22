#!/usr/bin/env bash

HOST="194.85.169.205:8000"

mkdir -p models; cd models

# BASE_MODEL="RF_model_YOLO_MbN/MbN2_416x416_t1"
# BASE_MODEL="RF_model_YOLO_Tiny/Tiny3_416x416_t1"
# BASE_MODEL="RF_model_YOLO_MbN/RF2_MbN2_384x416_t1"
# BASE_MODEL="RF_model_YOLO_MbN/RF_MbN2_384x416_t1"
# BASE_MODEL="RF_model_YOLO_Tiny/Tiny3_384x416_t1"
# BASE_MODEL="RF_model_YOLO_MbN_new/RF_MbN2_256x320_t1"
# BASE_MODEL="RF_model_YOLO_Tiny_new/Tiny3_256x320_t1"
BASE_MODEL="RF_model_YOLO_Tiny_new_v2/Tiny3_256x320_t1"

FP16_MODEL="$BASE_MODEL"_FP16
FP32_MODEL="$BASE_MODEL"_FP32

BASE_URL="http://$HOST/test_data"

wget "$BASE_URL/$BASE_MODEL.json" -O "model.json"

wget "$BASE_URL/$FP16_MODEL.bin" -O "model.bin"
wget "$BASE_URL/$FP16_MODEL.mapping" -O "model.mapping"
wget "$BASE_URL/$FP16_MODEL.xml" -O "model.xml"

# wget -N "$BASE_URL/$FP32_MODEL.bin"
# wget -N "$BASE_URL/$FP32_MODEL.mapping"
# wget -N "$BASE_URL/$FP32_MODEL.xml"
