#!/usr/bin/env bash

BACKBONE=ir50
LOSS_MODEL=qcface
DATA_DIR=./data/MS1MV3/train,./data/MS1MV3/valid # list of data root path separated with ','
WEIGHT_PATH=./qcface.pth # pth path
OUTPUT_DIR=./runs/${LOSS_MODEL}-${BACKBONE}

mkdir -p ${OUTPUT_DIR}

python train.py --arch ${BACKBONE} \
                --loss_model ${LOSS_MODEL} \
                --phase norm \
		        --data_dirs ${DATA_DIR} \
                --workers 16 \
                --epochs 12 \
                --start-epoch 0 \
                --batch-size 512 \
                --lr 0.01 \
                --momentum 0.9 \
                --weight-decay 5e-4 \
                --lr-drop-epoch 5 8 10 \
                --lr-drop-ratio 0.1 \
                --print-freq 100 \
                --pth-save-fold ${OUTPUT_DIR} \
                --pth-save-epoch 1 \
                --embed_dims 512 \
                --lambda_g 1.0 \
                --vis_mag 1 2>&1 | tee ${OUTPUT_DIR}/output.log
