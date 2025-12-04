import torch
import numpy as np
from tqdm import tqdm
import data_utils
import argparse
import pandas as pd
import evaluate_helper
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from face_model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--data_root", type=str, default='/data/data/faces/IJB/IJB_S')
    parser.add_argument('--backbone', type=str, default='ir18')
    parser.add_argument('--loss_model', type=str, default='qcface')
    parser.add_argument('--embed_dims', type=int, default=512)
    parser.add_argument('--train_data', type=str, default='casia')
    parser.add_argument('--weight_path', type=str, default='ir101_webface4m')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', default=8, type=int, help='')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--fuse_match_method', type=str, default='pre_norm_vector_add_cos',
                        choices=('pre_norm_vector_add_cos'))
    parser.add_argument('--save_features', action='store_true')

    args = parser.parse_args()

    # load model
    if args.loss_model == "magface":
        model = load_magface_model(args.weight_path, args.backbone, args.embed_dims)
    elif args.loss_model == "facenet":
        model = load_facenet_model(args.weight_path, args.backbone, args.embed_dims)
    else:
        if args.train_data=="casia":
            num_classes = 10572
        elif args.train_data=="msmv3":
            num_classes = 93431
        model = load_pretrained_model(args.weight_path, args.backbone, args.loss_model, args.embed_dims, num_classes)
    model.to('cuda:{}'.format(args.gpu))

    # make result save root
    save_root = './result/IJBS/{}-{}'.format(args.backbone, args.loss_model)
    os.makedirs(save_root, exist_ok=True)
    image_path_df = pd.read_csv('./image_list_mtcnn_fail_skipped_v1.csv', index_col=0)
    all_image_paths = image_path_df['path'].apply(lambda x:os.path.join(args.data_root, x)).tolist()

    num_partition = 100
    dataset_split = np.array_split(all_image_paths, num_partition)

    print('total {} images'.format(len(all_image_paths)))
    all_features = []
    for partition_idx in tqdm(range(num_partition)):

        image_paths = list(dataset_split[partition_idx])
        dataloader = data_utils.prepare_imagelist_dataloader(image_paths, batch_size=args.batch_size, num_workers=args.num_workers)

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()

        features = []
        norms = []
        prev_max_idx = 0
        with torch.no_grad():
            for iter_idx, (img, idx) in enumerate(dataloader):
                assert idx.max().item() > prev_max_idx
                prev_max_idx = idx.max().item()  # order shifting by dataloader checking
                if iter_idx % 100 == 0:
                    print(f"{iter_idx} / {len(dataloader)} done")
                feature = model(img.to("cuda:0"))

                if isinstance(feature, tuple) and len(feature) == 2:
                    feature, norm = feature
                    features.append(feature.cpu().numpy())
                    norms.append(norm.cpu().numpy())
                else:
                    features.append(feature.cpu().numpy())

        features = np.concatenate(features, axis=0)
        if args.save_features:
            save_path = os.path.join(save_root, 'feature_extracted/ijbs_pred_{}-{}_{}.npy'.format(args.backbone, args.loss_model, partition_idx))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, features)

        if len(norms) > 0:
            norms = np.concatenate(norms, axis=0)
            if args.save_features:
                save_path = os.path.join(save_root, 'feature_extracted/ijbs_pred_{}-{}_norm_{}.npy'.format(args.backbone, args.loss_model, partition_idx))
                np.save(save_path, norms)

        if args.fuse_match_method == 'pre_norm_vector_add_cos':
            features = features * norms
        all_features.append(features)
    all_features = np.concatenate(all_features, axis=0)

    # prepare savedir
    os.makedirs(os.path.join(save_root, 'eval_result'), exist_ok=True)
    # evaluate
    evaluate_helper.run_eval_with_features(save_root=save_root,
                                    features=all_features,
                                    image_paths=all_image_paths,
                                    get_retrievals=True,
                                    fuse_match_method=args.fuse_match_method)
