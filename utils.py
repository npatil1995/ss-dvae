import json
from pathlib import Path
from typing import Union, Any, List, Dict

import numpy as np
from scipy import sparse

import torch
import torch.nn.functional as F


def load_sparse(input_fname):
    return sparse.load_npz(input_fname).tocsr()

def load_json(fpath: Union[Path, str]) -> Any:
    with open(fpath) as infile:
        return json.load(infile)


def save_json(obj: Any, fpath: Union[Path, str]):
    with open(fpath, "w") as outfile:
        return json.dump(obj, outfile)


def save_topics(sorted_topics, inv_vocab, fpath, n=100):
    """
    Save topics to disk
    """
    with open(fpath, "w") as outfile:
        for topic_idx in sorted_topics:
            topic = [inv_vocab[i] for i in topic_idx[:n]]
            outfile.write(" ".join(topic) + "\n")


class NPMI:
    def __init__(
        self,
        bin_ref_counts: Union[np.ndarray, sparse.spmatrix],
        vocab: Dict[str, int] = None,
    ):
        assert bin_ref_counts.max() == 1
        self.bin_ref_counts = bin_ref_counts
        if sparse.issparse(self.bin_ref_counts):
            self.bin_ref_counts = self.bin_ref_counts.tocsc()
        self.npmi_cache = {} # calculating NPMI is somewhat expensive, so we cache results
        self.vocab = vocab

    def compute_npmi(
        self,
        beta: np.ndarray = None,
        topics: Union[np.ndarray, List] = None,
        vocab: Dict[str, int] = None,
        n: int = 10
    ) -> np.ndarray:
        """
        Compute NPMI for an estimated beta (topic-word distribution) parameter using
        binary co-occurence counts from a reference corpus

        Supply `vocab` if the topics contain terms that first need to be mapped to indices
        """
        if beta is not None and topics is not None:
            raise ValueError(
                "Supply one of either `beta` (topic-word distribution array) "
                "or `topics`, a list of index or word lists"
            )
        if vocab is None and any([isinstance(idx, str) for idx in topics[0][:n]]):
            raise ValueError(
                "If `topics` contains terms, not indices, you must supply a `vocab`"
            )
    
        if beta is not None:
            topics = np.flip(beta.argsort(-1), -1)[:, :n]
        if topics is not None:
            topics = [topic[:n] for topic in topics]
        if vocab is not None:
            assert(len(vocab) == self.bin_ref_counts.shape[1])
            topics = [[vocab[w] for w in topic[:n]] for topic in topics]

        num_docs = self.bin_ref_counts.shape[0]
        npmi_means = []
        for indices in topics:
            npmi_vals = []
            for i, idx_i in enumerate(indices):
                for idx_j in indices[i+1:]:
                    ij = frozenset([idx_i, idx_j])
                    try:
                        npmi = self.npmi_cache[ij]
                    except KeyError:
                        col_i = self.bin_ref_counts[:, idx_i]
                        col_j = self.bin_ref_counts[:, idx_j]
                        c_i = col_i.sum()
                        c_j = col_j.sum()
                        if sparse.issparse(self.bin_ref_counts):
                            c_ij = col_i.multiply(col_j).sum()
                        else:
                            c_ij = (col_i * col_j).sum()
                        if c_ij == 0:
                            npmi = 0.0
                        else:
                            npmi = (
                                (np.log(num_docs) + np.log(c_ij) - np.log(c_i) - np.log(c_j)) 
                                / (np.log(num_docs) - np.log(c_ij))
                            )
                        self.npmi_cache[ij] = npmi
                    npmi_vals.append(npmi)
            npmi_means.append(np.mean(npmi_vals))

        return np.array(npmi_means)


def compute_npmi(sorted_topics: np.ndarray, bin_ref_counts: np.ndarray, n: int = 10) -> np.ndarray:
    """
    Compute NPMI for an estimated beta (topic-word distribution) parameter using
    binary co-occurence counts from a reference corpus
    """
    num_docs = bin_ref_counts.shape[0]

    sorted_topics = sorted_topics[:, :n]

    npmi_means = []
    for indices in sorted_topics:
        npmi_vals = []
        for i, index1 in enumerate(indices):
            for index2 in indices[i+1:n]:
                col1 = bin_ref_counts[:, index1]
                col2 = bin_ref_counts[:, index2]
                c1 = col1.sum()
                c2 = col2.sum()
                if sparse.issparse(bin_ref_counts):
                    c12 = col1.multiply(col2).sum()
                else:
                    c12 = (col1 * col2).sum()

                if c12 == 0:
                    npmi = 0.0
                else:
                    npmi = (
                        (np.log(num_docs) + np.log(c12) - np.log(c1) - np.log(c2)) 
                        / (np.log(num_docs) - np.log(c12))
                    )
                npmi_vals.append(npmi)
        npmi_means.append(np.mean(npmi_vals))

    return np.array(npmi_means)


def compute_tu(topics, n=10):
    """
    Topic uniqueness measure from https://www.aclweb.org/anthology/P19-1640.pdf
    """
    tu_results = []
    for topics_i in topics:
        w_counts = 0
        for w in topics_i[:n]:
            w_counts += 1 / np.sum([w in topics_j[:n] for topics_j in topics]) # count(k, l)
        tu_results.append((1 / n) * w_counts)
    return tu_results


def compute_tr(topics, n=10):
    """
    Compute topic redundancy score from 
    https://jmlr.csail.mit.edu/papers/volume20/18-569/18-569.pdf
    """
    tr_results = []
    k = len(topics)
    for i, topics_i in enumerate(topics):
        w_counts = 0
        for w in topics_i[:n]:
            w_counts += np.sum([w in topics_j[:n] for j, topics_j in enumerate(topics) if j != i]) # count(k, l)
        tr_results.append((1 / (k - 1)) * w_counts)
    return tr_results


def compute_topic_exclusivity(beta, n=20):
    """
    Compute topic exclusivity, cited in https://arxiv.org/pdf/2010.12626.pdf
    """
    raise NotImplementedError()


def compute_to(topics, n=10, multiplier=2, return_overlaps=False):
    """
    A sensible overlap / redundancy measure. Words from a topic
    are only counted once per "edge"

    Basic algorithm creates a de-duplicated adjacency matrix:
    for each topic A_i, sorted by total number of overlaps:
        create set of sets S = {S_{ij} = A_i \cap A_j st. j=i+1,...,k}
        sort sets in S by their cardinality in descending order
        initialize a set W = {}
        For each S_{ij}' in S:
            if words are not already part of an edge, i.e., |W \cap S_{ij}'| is 0:
               create an edge between A_i and A_j with weight w = |S_{ij}'|
               augment the list of words used in an edge, W = W \cup S_{ij}'
    
    Then, sum the number of edges, weighted by some function of that number
    """
    k = len(topics)
    overlap_counts = np.zeros((k, k), dtype=int)
    overlap_dedup = np.zeros((k, k), dtype=int)
    overlap_words = {}

    # first count all the overlaps between topics
    for i, topic_i in enumerate(topics):
        for j, topic_j in enumerate(topics[i+1:], start=i+1):
            words_ij = set(topic_i[:n]) & set(topic_j[:n])
            overlap_counts[[i, j], [j, i]] = len(words_ij)
            overlap_words[frozenset([i, j])] = words_ij

    # sort topics by those with most overlaps
    sort_idx = overlap_counts.sum(0).argsort()[::-1]
    overlap_counts = overlap_counts[sort_idx, :][:, sort_idx]
    for i, counts in enumerate(overlap_counts):
        counted_words = set()
        start = i + 1
        for j in (counts[start:].argsort()[::-1] + start):
            words_ij = overlap_words[frozenset([i, j])]
            if overlap_counts[i, j] > 0 and len(counted_words & words_ij) == 0:
                overlap_dedup[i, j] = overlap_counts[i, j]
                counted_words |= words_ij

    # how many 1-word n-topic overlaps are equivalent to an n-word 1-topic overlap?
    increments = np.linspace(1/multiplier, n, num=n)
    redundancy = increments[overlap_dedup[overlap_dedup > 0] - 1].sum() / (n * (k - 1))
    if return_overlaps:
        redundancy = redundancy, overlap_dedup
    return redundancy

# ***************************************************************************
import torch
from torch.autograd import Variable


def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())


def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max

def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.

    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy