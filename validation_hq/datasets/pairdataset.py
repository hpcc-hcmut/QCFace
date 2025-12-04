import re
import typing

from .singledataset import SingleDataset
import numpy as np


def get_prefix(filenames, from_back):
    return '/'.join(filenames[0].split('/')[:-from_back])


def generate_pairs(labels, n=5, limit=10):
    distinct_labels = np.unique(labels).astype(int)
    genuine_pairs = []
    imposter_pairs = []
    for label in distinct_labels:
        genuine_idxs = np.argwhere(labels == label).squeeze()
        lgi = genuine_idxs.size
        if lgi > limit:
            genuine_idxs = np.random.choice(genuine_idxs, (limit,), replace=False)
            lgi = limit
        imposter_idxs = np.argwhere(labels != label).squeeze()
        for i in range(lgi):
            for j in range(i + 1, lgi):
                genuine_pairs.append([genuine_idxs[i], genuine_idxs[j]])
                imposters = np.random.randint(0, imposter_idxs.shape[0], (2, n))
                for k in range(n):
                    imposter_pairs.append([genuine_idxs[i], imposter_idxs[imposters[0, k]]])
                    imposter_pairs.append([genuine_idxs[j], imposter_idxs[imposters[1, k]]])

    genuine_pairs = np.array(genuine_pairs)
    imposter_pairs = np.array(imposter_pairs)
    pairs = np.vstack([genuine_pairs, imposter_pairs])
    matches = np.vstack([np.ones((genuine_pairs.shape[0], 1)), np.zeros((imposter_pairs.shape[0], 1))]).squeeze()
    idxs = np.arange(0, matches.shape[0], 1)
    np.random.shuffle(idxs)
    return pairs[idxs], matches[idxs]


def _get_pairs(root_dir, filenames, pairs_path, db):
    pms = np.loadtxt(pairs_path, delimiter=' ')
    if db=="agedb":
        db = "agedb_30"
    elif db=="cfp":
        db = "cfp_fp"
    pairs = []
    for a, b, m in pms:
        filename_a = f'{root_dir}/{db}/imgs/{int(a)}.jpg'
        filename_b = f'{root_dir}/{db}/imgs/{int(b)}.jpg'
        idx_a = np.where(filename_a == filenames)[0][0]
        idx_b = np.where(filename_b == filenames)[0][0]
        pairs.append([idx_a, idx_b])
    matches = pms[:, -1]
    return np.array(pairs), matches



def pair(root_dir, filenames, ids, pairs_path, db):
    pairs = None
    matches = None
    if pairs_path is None:
        pairs, matches = generate_pairs(ids)
    else:
        pairs, matches = _get_pairs(root_dir, filenames, pairs_path, db)
    return pairs, matches


class PairDataset:
    def __init__(self, single_dataset: typing.Union[SingleDataset, tuple], root_dir, pairs_file=None, pairs_function=pair):
        """
        Generates a dataset of genuine and imposter pairs with the underlying single dataset
        :param single_dataset: the dataset to use for generating pairs. Can also be the arguments for constructing a
        SingleDataset.
        :param pairs_file: path to the pairing file which defines all genuine and imposter pairs.
        :param pairs_function: the function which reads the pairs_file and converts it into indices for the single_images.
        """
        if isinstance(single_dataset, tuple):
            single_dataset = SingleDataset(*single_dataset)
        self.embeddings = single_dataset.embeddings
        self.filenames = single_dataset.filenames
        self.db = single_dataset.db
        self.model = single_dataset.model
        self.ids = single_dataset.ids
        self.pairs, self.matches = pairs_function(self.filenames, self.ids, pairs_file, self.db)
