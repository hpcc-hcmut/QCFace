BACKBONE=ir50
LOSS_MODEL=qcface
TRAIN_DATA=msmv3
DATA_DIR=./face-dataset/face_evaluation_data/ijb
WEIGHT_PATH=./face-models/qcface.pth
BATCH_SIZE=512
DEVICE_ID=0
SIMILARITY_METHOD=qmf # cosine, qmf (qmagface), euclid


echo ${LOSS_MODEL}-${BACKBONE}

echo Evaluating IJBB...
python validate_IJB_BC.py --dataset_name IJBB \
                          --data_root ${DATA_DIR} \
                          --backbone ${BACKBONE} \
                          --loss_model ${LOSS_MODEL} \
                          --embed_dims 512 \
                          --train_data ${TRAIN_DATA} \
                          --weight_path ${WEIGHT_PATH} \
                          --gpu ${DEVICE_ID} --batch_size ${BATCH_SIZE} --num_workers 16 \
                          --use_flip_test True --similarity_method ${SIMILARITY_METHOD}

echo Evaluating IJBC...
python validate_IJB_BC.py --dataset_name IJBC \
                          --data_root ${DATA_DIR} \
                          --backbone ${BACKBONE} \
                          --loss_model ${LOSS_MODEL} \
                          --embed_dims 512 \
                          --train_data ${TRAIN_DATA} \
                          --weight_path ${WEIGHT_PATH} \
                          --gpu ${DEVICE_ID} --batch_size ${BATCH_SIZE} --num_workers 16 \
                          --use_flip_test True --similarity_method ${SIMILARITY_METHOD}
