import argparse
import os
import tqdm
from similarity.cosine import Cosine
from similarity.qmagface import QMagFace
from similarity.euclid import Euclid
from datasets.pairdataset import PairDataset
from metrics import evaluate_metrics

def str2list(v):
    if ',' in v:
        return v.split(',')
    else:
        return [v]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tinyface')

    parser.add_argument('--train_db', type=str, default=None)
    parser.add_argument('--alpha', default=None, type=float)
    parser.add_argument('--beta', default=None, type=float)
    parser.add_argument('--data_root', type=str) 
    parser.add_argument('--feature_root', type=str)
    parser.add_argument('--pairs_root', type=str)
    parser.add_argument('--test_db', type=str2list)
    parser.add_argument('--result_dir', type=str)

    args = parser.parse_args()

    if args.alpha is not None and args.beta is not None:
        qmf = QMagFace(args.alpha, args.beta)
    elif args.train_db is not None:
        qmf = QMagFace()
        train_pds = PairDataset((os.path.join(args.feature_root, args.train_db), args.train_db), root_dir=args.data_root)
        qmf.train(train_pds.embeddings, train_pds.pairs, train_pds.matches)
        print(f'Found parameters alpha = {qmf.alpha} beta = {qmf.beta}')
    else:
        raise Exception('QMagFace cannot function without training or precomputed parameters.')

    qmf_accs = []
    cos_accs = []
    cos_auc_acc = []
    qmf_auc_acc = []
    for db in tqdm.tqdm(args.test_db):
        roc_dir = os.path.join(args.result_dir, db)
        if not os.path.exists(roc_dir):
            os.makedirs(roc_dir)

        pairs_file = os.path.join(args.pairs_root, f'{db}_pairs.list')
        pds = PairDataset((os.path.join(args.feature_root, db), db), root_dir=args.data_root, pairs_file=pairs_file)
        qmf_scores = qmf.similarity(pds.embeddings, pds.pairs)
        cos_scores = Cosine.similarity(pds.embeddings, pds.pairs)
        qmf_metrics = evaluate_metrics(qmf_scores, pds.matches, ['acc', 'acc_std', 'auc'], roc_name_path=os.path.join(roc_dir, f"qmag_{db}.txt"))
        cos_metrics = evaluate_metrics(cos_scores, pds.matches, ['acc', 'acc_std', 'auc'], roc_name_path=os.path.join(roc_dir, f"cos_{db}.txt"))
        qmf_accs.append(f"{qmf_metrics['acc']:.5f}+-{qmf_metrics['acc_std']:.5f}")
        cos_accs.append(f"{cos_metrics['acc']:.5f}+-{cos_metrics['acc_std']:.5f}")
        cos_auc_acc.append(cos_metrics['auc'])
        qmf_auc_acc.append(qmf_metrics['auc'])

    result = "db\t\tQMagFace\t\tCosine\t\t\tCosAUC\t\t\tQMFAUC\n"
    for db, qmf_acc, cos_acc, cos_auc, qmf_auc in zip(args.test_db, qmf_accs, cos_accs, cos_auc_acc, qmf_auc_acc):
        if len(db) > 6:      
            result += f"{db}\t{qmf_acc}\t{cos_acc}\t{cos_auc}\t{qmf_auc}\n"
        else:
            result += f"{db}\t\t{qmf_acc}\t{cos_acc}\t{cos_auc}\t{qmf_auc}\n"

    with open(os.path.join(args.result_dir, "results.txt"), 'w') as fw:
        fw.write(result)
        fw.close()
    
    print(result)

    ################################ FACENET ################################
    # euclid_accs = []
    # euclid_auc_acc = []
    # for db in tqdm.tqdm(args.test_db):
    #     roc_dir = os.path.join(args.result_dir, db)
    #     if not os.path.exists(roc_dir):
    #         os.makedirs(roc_dir)

    #     pairs_file = os.path.join(args.pairs_root, f'{db}_pairs.list')
    #     pds = PairDataset((os.path.join(args.feature_root, db), db), root_dir=args.data_root, pairs_file=pairs_file)
    #     euclid_scores = Euclid.similarity(pds.embeddings, pds.pairs)
    #     euclid_metrics = evaluate_metrics(euclid_scores, pds.matches, ['acc', 'acc_std', 'auc'], roc_name_path=os.path.join(roc_dir, f"euclid_{db}.txt"))
    #     euclid_accs.append(f"{euclid_metrics['acc']:.5f}+-{euclid_metrics['acc_std']:.5f}")
    #     euclid_auc_acc.append(euclid_metrics['auc'])

    # result = "db\t\tEuclid\t\t\tAUC\n"
    # for db, euclid_acc, euclid_auc in zip(args.test_db, euclid_accs, euclid_auc_acc):
    #     if len(db) > 6:      
    #         result += f"{db}\t{euclid_acc}\t{euclid_auc}\n"
    #     else:
    #         result += f"{db}\t\t{euclid_acc}\t{euclid_auc}\n"

    # with open(os.path.join(args.result_dir, "results.txt"), 'w') as fw:
    #     fw.write(result)
    #     fw.close()
    
    # print(result)
