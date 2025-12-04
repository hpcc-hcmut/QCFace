BACKBONE=ir18
LOSS_MODEL=qcface
DATA_DIR=./data/MS1MV3/train,./data/MS1MV3/valid # list of data root path separated with ','
OUTPUT_DIR=./runs/${LOSS_MODEL}-${BACKBONE}

mkdir -p ${OUTPUT_DIR}/vis/

python train.py --arch ${BACKBONE} \
                --loss_model ${LOSS_MODEL} \
                --phase norm \
                --data_dirs ${DATA_DIR} \
                --workers 4 \
                --epochs 25 \
                --start-epoch 0 \
                --batch-size 128 \
                --lr 0.1 \
                --momentum 0.9 \
                --weight-decay 5e-4 \
                --lr-drop-epoch 10 18 22 \
                --lr-drop-ratio 0.1 \
                --print-freq 100 \
                --pth-save-fold ${OUTPUT_DIR} \
                --pth-save-epoch 1 \
                --embed_dims 512 \
                --lambda_g 1.0 \
                --node_rank 0 \
                --gpus_per_node 2 \
                --num_nodes 1 \
                --master_addr 172.21.0.2 \
                --socket eth0 \
                --port 12355 \
                --vis_mag 1 2>&1 | sudo tee ${OUTPUT_DIR}/output.log
