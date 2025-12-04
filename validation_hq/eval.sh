BACKBONE=ir50
LOSS_MODEL=qcface
TRAIN_DATA=msmv3
DATA_DIR=./face_evaluation_data
FEATURE_DIR=./hq_face_feature # feature root path
WEIGHT_PATH=./face-models/qcface.pth
RESULT_DIR="./results/${LOSS_MODEL}"
BATCH_SIZE=512
DEVICE_ID=0

echo ${LOSS_MODEL}

echo Extract feature from AdienceGender dataset...
python genfeature.py --source_dir "${DATA_DIR}/AdienceGender/aligned" \
                     --result_dir ${FEATURE_DIR} \
                     --backbone ${BACKBONE} \
                     --loss_model ${LOSS_MODEL} \
                     --embed_dims 512 \
                     --train_data ${TRAIN_DATA} \
                     --weight_path ${WEIGHT_PATH} \
                     --gpu ${DEVICE_ID} --batch_size ${BATCH_SIZE} --num_workers 16

echo Extract feature from AgeDB-30 dataset...
python genfeature.py --source_dir "$DATA_DIR/agedb_30/imgs" \
                     --result_dir ${FEATURE_DIR} \
                     --backbone ${BACKBONE} \
                     --loss_model ${LOSS_MODEL} \
                     --embed_dims 512 \
                     --train_data ${TRAIN_DATA} \
                     --weight_path ${WEIGHT_PATH} \
                     --gpu ${DEVICE_ID} --batch_size ${BATCH_SIZE} --num_workers 16

echo Extract feature from CFP-FP dataset...
python genfeature.py --source_dir "$DATA_DIR/cfp_fp/imgs" \
                     --result_dir ${FEATURE_DIR} \
                     --backbone ${BACKBONE} \
                     --loss_model ${LOSS_MODEL} \
                     --embed_dims 512 \
                     --train_data ${TRAIN_DATA} \
                     --weight_path ${WEIGHT_PATH} \
                     --gpu ${DEVICE_ID} --batch_size ${BATCH_SIZE} --num_workers 16

echo Extract feature from LFW dataset...
python genfeature.py --source_dir "$DATA_DIR/lfw/imgs" \
                     --result_dir ${FEATURE_DIR} \
                     --backbone ${BACKBONE} \
                     --loss_model ${LOSS_MODEL} \
                     --embed_dims 512 \
                     --train_data ${TRAIN_DATA} \
                     --weight_path ${WEIGHT_PATH} \
                     --gpu ${DEVICE_ID} --batch_size ${BATCH_SIZE} --num_workers 16

echo Extract feature from CALFW dataset...
python genfeature.py --source_dir "$DATA_DIR/calfw/imgs" \
                     --result_dir ${FEATURE_DIR} \
                     --backbone ${BACKBONE} \
                     --loss_model ${LOSS_MODEL} \
                     --embed_dims 512 \
                     --train_data ${TRAIN_DATA} \
                     --weight_path ${WEIGHT_PATH} \
                     --gpu ${DEVICE_ID} --batch_size ${BATCH_SIZE} --num_workers 16

echo Extract feature from CPLFW dataset...
python genfeature.py --source_dir "$DATA_DIR/cplfw/imgs" \
                     --result_dir ${FEATURE_DIR} \
                     --backbone ${BACKBONE} \
                     --loss_model ${LOSS_MODEL} \
                     --embed_dims 512 \
                     --train_data ${TRAIN_DATA} \
                     --weight_path ${WEIGHT_PATH} \
                     --gpu ${DEVICE_ID} --batch_size ${BATCH_SIZE} --num_workers 16

echo Extract feature from XQLFW dataset...
python genfeature.py --source_dir "$DATA_DIR/xqlfw/imgs" \
                     --result_dir ${FEATURE_DIR} \
                     --backbone ${BACKBONE} \
                     --loss_model ${LOSS_MODEL} \
                     --embed_dims 512 \
                     --train_data ${TRAIN_DATA} \
                     --weight_path ${WEIGHT_PATH} \
                     --gpu ${DEVICE_ID} --batch_size ${BATCH_SIZE} --num_workers 16

echo Evaluation...
python eval.py --train_db "AdienceGender" \
               --data_root "./data/face-dataset/face_evaluation_data" \
               --feature_root "$FEATURE_DIR/$LOSS_MODEL" \
               --pairs_root "$DATA_DIR/pairlists" \
               --test_db agedb_30,cfp_fp,lfw,calfw,cplfw,xqlfw \
               --result_dir $RESULT_DIR && rm -rf ${FEATURE_DIR}

# For FaceNet
# echo Evaluation...
# python eval.py --data_root "./data/face-dataset/face_evaluation_data" \
#                --feature_root "$FEATURE_DIR/$LOSS_MODEL" \
#                --pairs_root "$DATA_DIR/pairlists" \
#                --test_db agedb_30,cfp_fp,lfw,calfw,cplfw,xqlfw \
#                --result_dir $RESULT_DIR && rm -rf ${FEATURE_DIR}
