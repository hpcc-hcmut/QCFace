python3 validate_tinyface.py --data_root "/media/phuong/data/phuong/data/face_evaluation_data/tinyface" \
                             --gpu 0 --batch_size 512 --num_workers 8 \
                             --backbone ir18 \
                             --loss_model qcface \
                             --embed_dims 512 \
                             --train_data casia \
                             --weight_path "/home/phuong/Workspace/QMagFace/_models/facenet/best_state.pth"
