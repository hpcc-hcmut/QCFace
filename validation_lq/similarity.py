import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

#------------- CONFIG alpha and beta for QMagFace --------------
# # For IR100
# QMF_HP = dict(
#     magface = (0.21816225563519392, 0.32816204253326076),
#     adaface = (0, 0),
#     qcface = (0.021486797990885483, 0.030095898385268535)
# )

# # IR50
# QMF_HP = dict(
#     magface = (0.4734050630153164, 0.6396150228991917),
#     adaface = (0, 0),
#     qcface = (0.05298834571695404, 0.08483867351227943)
# )

# For IR18
QMF_HP = dict(
    magface = (0.020954943801255458, 4.8051245064235626e-30),
    adaface = (0, 0),
    qcface = (0.02323473949482023, 0.02969502085476521)
)

CUTOFF_VALUE = 0

def verification_similarity(feat1, feat2, sim_method, model_name=None):
    if sim_method in ["cosine", "qmf"]:
        norm_feat1, norm1 = normalize(feat1, return_norm=True)
        norm_feat2, norm2 = normalize(feat2, return_norm=True)
        sim = np.sum(norm_feat1 * norm_feat2, axis=-1)
        if sim_method=="cosine":
            return sim
        else:
            q = np.min(np.stack([norm1, norm2], axis=1), axis=1)
            alpha, beta = QMF_HP[model_name]
            weight = beta * sim - alpha
            weight[weight > CUTOFF_VALUE] = 0
            sim = weight * q + sim
            return sim
    else:
        distance = np.linalg.norm(feat1 - feat2, axis=1)
        return (distance.max() - distance) / (distance.max() - distance.min() + 1e-10)


def identification_similarity(query_feats, gallery_feats, sim_method, model_name=None):
    if sim_method in ["cosine", "qmf"]:    
        norm_query_feats, query_norm = normalize(query_feats, return_norm=True)
        norm_gallery_feats, gallery_norm = normalize(gallery_feats, return_norm=True)
        sim = np.dot(norm_query_feats, norm_gallery_feats.T)
        if sim_method=="cosine":
            return sim
        else:
            gallery_norm, query_norm = np.meshgrid(gallery_norm, query_norm)
            q = np.min(np.stack([query_norm, gallery_norm], axis=-1), axis=-1)
            alpha, beta = QMF_HP[model_name]
            weight = beta * sim - alpha
            weight[weight > CUTOFF_VALUE] = 0
            sim = weight * q + sim
            return sim
    else:
        distance = cdist(query_feats, gallery_feats, metric='euclidean')
        return (distance.max() - distance) / (distance.max() - distance.min() + 1e-10)
