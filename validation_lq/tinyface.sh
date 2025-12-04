BACKBONE=ir50
LOSS_MODEL=qcface
TRAIN_DATA=msmv3
DATA_DIR=./face_evaluation_data/tinyface # data root path
WEIGHT_PATH=./face-models/qcface.pth # pth path
BATCH_SIZE=512
DEVICE_ID=0
SIMILARITY_METHOD=qmf # cosine, qmf (qmagface), euclid

echo $LOSS_MODEL

python3 validate_tinyface.py --data_root ${DATA_DIR} \
                             --gpu ${DEVICE_ID} --batch_size ${BATCH_SIZE} --num_workers 16 \
                             --backbone ${BACKBONE} \
                             --loss_model ${LOSS_MODEL} \
                             --embed_dims 512 \
                             --train_data ${TRAIN_DATA} \
                             --weight_path ${WEIGHT_PATH} \
                             --similarity ${SIMILARITY_METHOD}