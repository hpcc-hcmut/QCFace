import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from face_model import *
import argparse
from tqdm import tqdm
import numpy as np
import torch
from dataloader import dataloader_generation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract HQ feature')
    # general
    parser.add_argument('--source_dir', "-s", default="/media/phuong/data/phuong/data/face_evaluation_data/agedb_30/imgs")
    parser.add_argument('--result_dir', "-r", default="/media/phuong/data/phuong/data/face_evaluation_data/agedb_30/results")
    parser.add_argument('--backbone', type=str, default='ir18')
    parser.add_argument('--loss_model', type=str, default='qcface')
    parser.add_argument('--embed_dims', type=int, default=512)
    parser.add_argument('--train_data', type=str, default='casia')
    parser.add_argument('--weight_path', type=str, default='ir101_webface4m')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch_size', default=128, type=int, help='')
    parser.add_argument('--num_workers', default=8, type=int, help='')
    args = parser.parse_args()

    # load model
    if args.loss_model == "facenet":
        model = load_facenet_model(args.weight_path, args.backbone, args.embed_dims)
    else:
        if args.train_data=="casia":
            num_classes = 10572
        elif args.train_data=="msmv3":
            num_classes = 93431
        model = load_pretrained_model(args.weight_path, args.backbone, args.loss_model, args.embed_dims, num_classes)
    model.to('cuda:{}'.format(args.gpu))

    dsname = args.source_dir.split('/')[-2]
    loader = dataloader_generation(dsname, args.source_dir, args.num_workers, args.batch_size)

    embeddings = torch.tensor([], dtype=torch.float32)

    with torch.no_grad():
        for imgs, path in tqdm(loader):
            emb = model(imgs.to("cuda:{}".format(args.gpu))).cpu().detach()
            embeddings = torch.cat([embeddings, emb], dim=0)
    
    embeddings = embeddings.numpy()
    filenames = np.array(loader.dataset.file_list)

    result_dir = os.path.join(args.result_dir, args.loss_model, dsname)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    np.save(os.path.join(result_dir, f'embeddings_{dsname}.npy'), embeddings)
    np.save(os.path.join(result_dir, f'filenames_{dsname}.npy'), filenames)
