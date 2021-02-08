#!/usr/bin/env bash

python src/train.py \
    -c common.yml \
    -p train.pretrained="s3://mlflow/artifacts/3/a310628bbdd3493bbbc8078b2da77465/artifacts/rf_signs_detection_Yolo4Tiny_best_map75.pth" \
        train.skip_eval_epochs=0 \
    $@
