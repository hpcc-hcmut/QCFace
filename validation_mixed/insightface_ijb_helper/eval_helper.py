import numpy as np
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import os
from sklearn.metrics import roc_curve, auc
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import csv
from similarity import verification_similarity as similarity

def read_template_media_list(path):
    #ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(int)
    medias = ijb_meta[:, 2].astype(int)
    return templates, medias


def read_template_pair_list(path):
    #pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(int)
    t2 = pairs[:, 1].astype(int)
    label = pairs[:, 2].astype(int)
    return t1, t2, label


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


def image2template_feature(img_feats=None, templates=None, medias=None, combine_func=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t, ) = np.where(templates == uqt)
        face_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m, ) = np.where(face_medias == u)
            if ct == 1:
                media_feats += [face_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_feats += [combine_func(face_feats[ind_m])]
        media_feats = np.array(media_feats)
        # template_feats[count_template] = combine_func(media_feats)
        template_feats[count_template] = np.sum(media_feats, 0)
        
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    
    return template_feats, unique_templates


def verification(template_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None,
                 similarity_method="cosine",
                 loss_model=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1), ))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = np.squeeze(template_feats[template2id[p1[s]]], 1)
        feat2 = np.squeeze(template_feats[template2id[p2[s]]], 1)

        score[s] = similarity(feat1, feat2, similarity_method, loss_model)
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats

def write_result(result_files, save_path, dataset_name, label):

    methods = []
    scores = []
    for file in result_files:
        methods.append(Path(file).parent.stem)
        scores.append(np.load(file))

    methods = np.array(methods)
    scores = dict(zip(methods, scores))
    colours = dict(zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
    #x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
    x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels] + ['AUC'])
    fig = plt.figure()
    for method in methods:
        fpr, tpr, _ = roc_curve(label, scores[method])
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr
        plt.plot(fpr,
                 tpr,
                 color=colours[method],
                 lw=1,
                 label=('[%s (AUC = %0.4f %%)]' % (method.split('-')[-1], roc_auc * 100)))
        tpr_fpr_row = []
        tpr_fpr_row.append("%s-%s" % (method, dataset_name))
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
        tpr_fpr_row.append('%.2f' % (roc_auc * 100))
        tpr_fpr_table.add_row(tpr_fpr_row)
    plt.xlim([10**-6, 0.1])
    plt.ylim([0.3, 1.0])
    plt.grid(linestyle='--', linewidth=3)
    plt.xticks(x_labels)
    plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on IJB')
    plt.legend(loc="lower right")
    #plt.show()
    fig.savefig(os.path.join(save_path, 'verification_auc.pdf'))
    print(tpr_fpr_table)

    # write to csv
    result = [tuple(filter(None, map(str.strip, splitline))) for line in str(tpr_fpr_table).splitlines()
                                                             for splitline in [line.split("|")] if len(splitline) > 1]
    with open(os.path.join(save_path, 'verification_result.csv'), 'w') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerows(result)
