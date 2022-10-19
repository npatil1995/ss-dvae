import logging
from os import stat
import warnings
from typing import Dict, Optional, Union, Tuple
from pathlib import Path

import typer
import numpy as np
import pandas as pd
import tqdm
import sys
import pickle
from scipy import sparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO, JitTraceMeanField_ELBO
from pyro.optim import Adam

from utils import NPMI, compute_to, compute_tu, load_json, save_topics, load_sparse


from model import ElboCalculation, VariationalAutoencoder

logger = logging.getLogger(__name__)

app = typer.Typer()


# class L1RegularizedTraceMeanField_ELBO(TraceMeanField_ELBO):
#     def __init__(self, *args, l1_params=None, l1_weight=1., **kwargs):
#         super().__init__(*args, **kwargs)
#         self.l1_params = l1_params
#         self.l1_weight = l1_weight
#
#     @staticmethod
#     def l1_regularize(param_names, weight):
#         params = torch.cat([pyro.param(p).view(-1) for p in param_names])
#         return weight * torch.norm(params, 1)
#
#
#     def loss_and_grads(self, model, guide, *args, **kwargs):
#         loss_standard = self.differentiable_loss(model, guide, *args, **kwargs)
#         loss = loss_standard + self.l1_regularize(self.l1_params, self.l1_weight)
#
#         loss.backward()
#         loss = loss.item()
#
#         pyro.util.warn_if_nan(loss, "loss")
#         return loss
#
#
# class CollapsedMultinomial(dist.Multinomial):
#     """
#     Equivalent to n separate `MultinomialProbs(probs, 1)`, where `self.log_prob` treats each
#     element of `value` as an independent one-hot draw (instead of `MultinomialProbs(probs, n)`)
#     """
#     def log_prob(self, value: torch.tensor) -> torch.tensor:
#         return ((self.probs + 1e-10).log() * value).sum(-1)
#
#
# class LinearSoftmax(nn.Linear):
#     """
#     Linear layer where the weights are first put through a softmax
#     """
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         # TODO: should we even allow a bias?
#         return F.linear(input, F.softmax(self.weight, dim=0), self.bias)


class Encoder(nn.Module):
    """
    Module that parameterizes the dirichlet distribution q(z|x)
    """
    def __init__(
            self,
            vocab_size: int,
            num_topics: int,
            embeddings_dim: int,
            hidden_dim: int,
            dropout: float,
        ):
        super().__init__()

        # setup linear transformations
        self.embedding_layer = nn.Linear(vocab_size, embeddings_dim)
        self.embed_drop = nn.Dropout(dropout)

        self.second_hidden_layer = hidden_dim > 0
        if self.second_hidden_layer:
            self.fc = nn.Linear(embeddings_dim, hidden_dim)
            self.fc_drop = nn.Dropout(dropout)
        #was self.alpha_layer = nn.Linear(hidden_dim or embeddings_dim, num_topics)

        self.alpha_layer = nn.Linear(hidden_dim or embeddings_dim, num_topics)

        # this matches NVDM / TF implementation, which does not use scale
        self.alpha_bn_layer = nn.BatchNorm1d(
            num_topics, eps=0.001, momentum=0.001, affine=True
        )
        self.alpha_bn_layer.weight.data.copy_(torch.ones(num_topics))
        self.alpha_bn_layer.weight.requires_grad = False

    def forward(self, x: torch.tensor) -> torch.tensor:
        embedded = F.relu(self.embedding_layer(x))
        embedded_do = self.embed_drop(embedded)

        hidden_do = embedded_do
        if self.second_hidden_layer:
            hidden = F.relu(self.fc(embedded_do))
            hidden_do = self.fc_drop(hidden)

        alpha = self.alpha_layer(hidden_do)
        alpha_bn = self.alpha_bn_layer(alpha)

        alpha_pos = torch.max(
            F.softplus(alpha_bn),
            torch.tensor(0.00001, device=alpha_bn.device)
        )

        return alpha_pos


class Decoder(nn.Module):
    """
    Module that parameterizes the obs likelihood p(x | z)
    """
    def __init__(
        self,
        vocab_size: int,
        num_topics: int, 
        bias_term: bool = True,
        softmax_beta: bool = False,
        beta_init: torch.tensor = None,
    ):
        super().__init__()

        if not softmax_beta:
            self.eta_layer = nn.Linear(num_topics, vocab_size, bias=bias_term)
        else:
            self.eta_layer = LinearSoftmax(num_topics, vocab_size, bias=bias_term)
        
        if beta_init is not None:
            self.eta_layer.weight.data.copy_(beta_init.T)

        # this matches NVDM / TF implementation, which does not use scale
        self.eta_bn_layer = nn.BatchNorm1d(
            vocab_size, eps=0.001, momentum=0.001, affine=True
        )
        self.eta_bn_layer.weight.data.copy_(torch.ones(vocab_size))
        self.eta_bn_layer.weight.requires_grad = False

    def forward(self, z: torch.tensor, bn_annealing_factor: float = 0.0) -> torch.tensor:
        eta = self.eta_layer(z)
        eta_bn = self.eta_bn_layer(eta)

        x_recon = (
            (bn_annealing_factor) * F.softmax(eta, dim=-1)
            + (1 - bn_annealing_factor) * F.softmax(eta_bn, dim=-1)
        )
        return x_recon
    
    @property
    def beta(self) -> np.ndarray:
        return self.eta_layer.weight.T.cpu().detach().numpy()


class DVAE(nn.Module):
    """
    Pytorch module for the Dirichlet-VAE
    """
    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        alpha_prior: float,
        embeddings_dim: int,
        hidden_dim: int,
        dropout: float,
        bias_term: bool = True,
        softmax_beta: bool = False,
        beta_init: torch.tensor = None,
        cuda: bool = True,
    ):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(
            vocab_size=vocab_size,
            num_topics=num_topics,
            embeddings_dim=embeddings_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            num_topics=num_topics,
            bias_term=bias_term,
            softmax_beta=softmax_beta,
            beta_init=beta_init,
        )

        if cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = cuda
        self.num_topics = num_topics
        self.alpha_prior = alpha_prior

    # define the model p(x|z)p(z)
    def model(
        self, x: torch.tensor,
        bn_annealing_factor: float = 1.0,
        kl_annealing_factor: float = 1.0
    ) -> torch.tensor:
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            # print("x shape",x.shape)
            # print(x.shape[0])
            # print(self.num_topics)
            # print("alpha",self.alpha_prior)

            alpha_0 = torch.ones(
                x.shape[0], self.num_topics, device=x.device
            ) * self.alpha_prior
            # print("alpha_0",alpha_0.shape)
            # print("alpha_0 value :", alpha_0)


            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("doc_topics", dist.Dirichlet(alpha_0))
            print("here in decoder")
            # print("z", z)
            # print(z.size())

            # decode the latent code z
            x_recon = self.decoder(z, bn_annealing_factor)
            # score against actual data
            pyro.sample("obs", CollapsedMultinomial(1, probs=x_recon), obs=x)

            return x_recon

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(
        self, x: torch.tensor,
        bn_annealing_factor: float = 1.0,
        kl_annealing_factor: float = 1.0
    ):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            print("hello")
            # use the encoder to get the parameters used to define q(z|x)
            z = self.encoder(x)
            print("here in encoder")
            # sample the latent code z
            with pyro.poutine.scale(None, kl_annealing_factor):
                pyro.sample("doc_topics", dist.Dirichlet(z))


def calculate_annealing_factor(
    current_batch: int,
    current_epoch: int,
    epochs_to_anneal: int,
    batches_per_epoch: int,
    min_af: float = 0.01,
) -> float:
    """
    Calculate annealing factor. Modified from pyro/examples/dmm.py
    """
    if epochs_to_anneal > 0:
        return min(
            1.0, # this hits when epochs_to_anneal > current_epoch
            min_af + (1.0 - min_af) * # when epochs_to_anneal <= current_epoch
            np.divide(
                (current_batch + current_epoch * batches_per_epoch + 1),
                (epochs_to_anneal * batches_per_epoch)
            )
        )
    else:
        return 0.0


def data_iterator(
    data: Union[np.ndarray, sparse.spmatrix],
    batch_size: int,
    num_batches: int
    ) -> torch.tensor:
    for i in range(num_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]
        if sparse.issparse(batch):
            batch = batch.toarray()
        yield torch.tensor(batch)


@app.command(help="Run a Dirichlet-VAE (with pytorch)")

#changed files for input and output
def run_dvae(
        input_dir: Optional[Path] = None,
        output_dir: Path = "E:\thesis papers temp\dvae-main\dvae-main\output",
        train_path: Path = "train_data_updated.npz",
        eval_path: Optional[Path] = "valid_data_updated.npz",
        vocab_path: Optional[Path] = "vocab_updated.json",
        
        topic_word_init_path: Optional[Path] = None, # initialize the topic-word distribution
        topic_word_prior_path: Optional[Path] = None, # use the topic-word distribution as a prior
        # TODO: add priors for document-topics
        num_topics: Optional[int] = None,
        to_dense: bool = True,

        encoder_embeddings_dim: int = 100,
        encoder_hidden_dim: int = 0, # setting to 0 turns off the second layer
        dropout: float = 0.25, # TODO: separate for enc/dec
        alpha_prior: float = 0.01,
        decoder_bias: bool = True,
        softmax_beta: bool = False,

        learning_rate: float = 0.001,
        topic_word_regularization: Optional[float] = None, 
        adam_beta_1: float = 0.9,
        adam_beta_2: float = 0.999,
        batch_size: int = 200,
        num_epochs: int = 200,
        epochs_to_anneal_bn: int = 0, # 0 is BN as usual; 1 turns BN off; >1 anneals from BN on to BN off
        epochs_to_anneal_kl: int = 100, # 0 throws error; 1 is standard KL; >1 anneals from 0 to 1 KL weight

        eval_words: int = 10,
        topic_words_to_save: int = 50,
        target_metric: Optional[str] = None,
        compute_eval_loss: bool = False,
        max_acceptable_overlap: Optional[int] = None,
        eval_step: int = 1,
        save_all_topics: bool = True,
        
        seed: int = 42,
        gpu: bool = False,
        jit: bool = True, # currently not used since it's caused some crashes
    ) -> Tuple[DVAE, Dict[str, float]]:

    # clear param store
    pyro.clear_param_store()
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    pyro.enable_validation(__debug__)

    if input_dir is not None:
        train_path = Path(input_dir,  train_path)
        eval_path = Path(input_dir, eval_path)
        vocab_path = Path(input_dir, vocab_path)

    if output_dir is None:
        raise ValueError("`output_dir` is not set")

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    if save_all_topics:
        Path(output_dir, "topics").mkdir(exist_ok=True, parents=True)

    # load the data
    # x_train = load_sparse(train_path).astype(np.float32)
    train_data = np.load("train_data_updated.npz")
    x_train = torch.from_numpy(train_data["train_data"]).float()
    x_train_labels = torch.from_numpy(train_data["target"])
    # print("targets: ",x_train_labels)

   #*****************************code updated here*************************************************
    topic_vocabulary = {
        "alt.atheism": 0,
        "comp.graphics": 1,
        "comp.os.ms-windows.misc": 2,
        "comp.sys.ibm.pc.hardware": 3,
        "comp.sys.mac.hardware": 4,
        "comp.windows.x": 5,
        "misc.forsale": 6,
        "rec.autos": 7,
        "rec.motorcycles": 8,
        "rec.sport.baseball": 9,
        "rec.sport.hockey": 10,
        "sci.crypt": 11,
        "sci.electronics": 12,
        "sci.med": 13,
        "sci.space": 14,
        "soc.religion.christian": 15,
        "talk.politics.guns": 16,
        "talk.politics.mideast": 17,
        "talk.politics.misc": 18,
        "talk.religion.misc": 19
        # "common_topic": 20
    }
    #create label projection matrix
    labels_corpus = []
    x_train_labels = x_train_labels.tolist()
    for label_id in x_train_labels:
        labels = []
        label_name = list(topic_vocabulary.keys())[list(topic_vocabulary.values()).index(label_id)]
        labels.append(label_name)
        # labels.append("common_topic")
        labels_corpus.append(labels)


    M = x_train.shape[0]
    K = len(topic_vocabulary)

    Lambda = np.zeros((M, K), dtype=float)
    # print(Lambda.shape)
    for m in range(M):
        for label in labels_corpus[m]:
            k = topic_vocabulary[label]
            Lambda[m, k] = 1.0

    # print("Labels", Lambda)

    train_data = []
    for i in range(len(x_train)):
        train_data.append([x_train[i], Lambda[i]])

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    # train_data_batch, labels_batch = next(iter(trainloader))
    # print(train_data_batch.shape)
    # print(labels_batch.shape)


    # for epoch in range(2):
    #     for step, (x, y) in enumerate(trainloader):
    #         # print("step",step)
    #         # print("data", x.shape)
    #         # print("labels", y.shape)


    # for x,y in zip(x_train,Lambda):
    #     print("X",x)
    #     print("Y",y)



    # def _multinomial_sample(p_vector, random_state=None):
    #     """
    #     sample a number from multinomial distribution
    #     :param p_vector: the probabilities
    #     :return: a int value
    #     """
    #     if random_state is not None:
    #         return random_state.multinomial(1, p_vector).argmax()
    #     return np.random.multinomial(1, p_vector).argmax()
    #
    # Z = []
    # for m in range(M):
    #     # print "self.Lambda[m]: ", self.Lambda[m]
    #     numerator_vector = Lambda[m] * alpha_prior
    #     p_vector = 1.0 * numerator_vector / sum(numerator_vector)
    #     # print p_vector
    #     # print "p_vector: ", p_vector
    #     # z_vector is a vector of a document,
    #     # just like [2, 3, 6, 0], which means this doc have 4 word and them generated
    #     # from the 2nd, 3rd, 6th, 0th topic, respectively
    #     z_vector = [_multinomial_sample(p_vector) for _ in range(len(W[m]))]
    #     Z.append(z_vector)

    # *****************************code updated here*************************************************

    # unique_targets = torch.unique(x_train_labels)
    # print(unique_targets)

    # x_train = sparse.csr_matrix(train_data["train_data"])
    if to_dense:
        x_train = x_train.astype(np.float32).toarray()
    n_train = x_train.shape[0]

    vocab_size = x_train.shape[1]


    beta_init, beta_prior = None, None
    if topic_word_init_path:
        beta_init = torch.tensor(np.load(topic_word_init_path))
    if topic_word_prior_path:
        raise NotImplementedError("Need to implement topic-word prior (may involve ditching L1 regularization for laplacian?)")
        beta_prior = torch.tensor(np.load(topic_word_prior_path))

    if eval_path is not None:
        valid_data = np.load("valid_data_updated.npz")
        # x_val = torch.from_numpy(valid_data["valid_data"]).float()
        x_val = sparse.csr_matrix(valid_data["valid_data"])
        # print(type(valid_data["valid_data"]))
        # print(type(x_val))
        # x_val = load_sparse(eval_path).astype(np.float32)
        if not compute_eval_loss and not to_dense:
            # x_val = x_val
            x_val = x_val.tocsc() # slight speedup
        if to_dense:
            #changed x_val = x_val.toarray().astype(np.uint8)
            x_val = x_val.toarray()
    else:
        compute_eval_loss = False # will be identical to train loss
        x_val, n_val = x_train, n_train

    n_val = x_val.shape[0]

    # load the vocabulary
    vocab = pickle.load(open("vocab_updated", "rb"))
    # vocab = load_json(vocab_path) if vocab_path else None
    inv_vocab = dict(zip(vocab.values(), vocab.keys()))

    # print("here")


    # # setup the VAE
    # vae = DVAE(
    #     vocab_size=vocab_size,
    #     num_topics=num_topics,
    #     alpha_prior=alpha_prior,
    #     embeddings_dim=encoder_embeddings_dim,
    #     hidden_dim=encoder_hidden_dim,
    #     dropout=dropout,
    #     bias_term=decoder_bias,
    #     softmax_beta=softmax_beta,
    #     beta_init=beta_init,
    #     cuda=gpu,
    # )

    # setup the optimizer
    adam_args = {
        "lr": learning_rate,
        "betas": (adam_beta_1, adam_beta_2), # from ProdLDA
    }
    optimizer = Adam(adam_args)

    # ******************************code updated here***************************************************
    print("num",num_topics)

    ssvae_model = VariationalAutoencoder(
        vocab_size=vocab_size,
        num_topics=num_topics,
        alpha_prior=alpha_prior,
        embeddings_dim=encoder_embeddings_dim,
        hidden_dim=encoder_hidden_dim,
        dropout=dropout,
        bias_term=decoder_bias,
        softmax_beta=softmax_beta,
        beta_init=beta_init,
        cuda=gpu,
    )

    for epoch in range(num_epochs):
        ssvae_model.train()
        for x,y in trainloader:
            elbo_calculation = ElboCalculation(x, y, ssvae_model)

    # **********************************code updated here*****************************************

    # setup the inference algorithm
    elbo = TraceMeanField_ELBO()
    if topic_word_regularization:
        elbo = L1RegularizedTraceMeanField_ELBO(
            l1_params=["decoder$$$eta_layer.weight", "decoder$$$eta_layer.bias"],
            l1_weight=topic_word_regularization
        )
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    train_elbo = []
    results_path = Path(output_dir, "results.csv")
    val_metrics = {
        "val_loss": np.inf,
        "npmi": 0,
        "tu": 0,
        "to": 0,
    }
    # print(type(x_val))
    # print(x_val.shape)
    npmi_scorer = NPMI((x_val > 0).astype(int))
    if max_acceptable_overlap is None:
        max_acceptable_overlap = float("inf")

    # training loop
    t = tqdm.trange(num_epochs, leave=True)
    train_batches = n_train // batch_size
    eval_batches = n_val // batch_size
    result_message = {}

    loss = []
    epoch_num = []

    for epoch in t:
        # initialize loss accumulator
        random_idx = np.random.choice(n_train, size=n_train, replace=False)
        x_train = x_train[random_idx] #shuffle
        # print("x_train", x_train.shape)

        epoch_loss = 0.
        for i, x_batch in enumerate(data_iterator(x_train, batch_size, train_batches)):
            print("x_batch",x_batch.shape)
            # if on GPU put mini-batch into CUDA memory


            if gpu:
                x_batch = x_batch.cuda()

            bn_af = calculate_annealing_factor(i, epoch, epochs_to_anneal_bn, train_batches)
            kl_af = calculate_annealing_factor(i, epoch, epochs_to_anneal_kl, train_batches)
            bn_af = torch.tensor(bn_af, device=x_batch.device)
            kl_af = torch.tensor(kl_af, device=x_batch.device)


            print("before epoch loss")
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x_batch, bn_af, kl_af)
            print("after epoch loss")

        # report training diagnostics
        epoch_loss /= n_train
        train_elbo.append(epoch_loss)

        loss.append(epoch_loss)
        epoch_num.append(epoch)




        # evaluate on the validation set
        result_message.update({"tr loss": f"{epoch_loss:0.1f}"})

        if epoch % eval_step == 0:
            
            # get loss
            val_loss = epoch_loss # set to training loss if not being calculated
            if compute_eval_loss:
                val_loss = 0.
                for x_batch in data_iterator(x_val, batch_size, eval_batches):
                    if gpu:
                        x_batch = x_batch.cuda()
                    val_loss += svi.evaluate_loss(x_batch, bn_af, kl_af)
                val_loss /= n_val

            # get beta and topic terms
            beta = vae.decoder.beta
            print("beta", beta.shape)
            topic_terms = np.flip(beta.argsort(-1), -1)[:, :topic_words_to_save]

            # compute topic-uniqueness & topic overlap
            to, overlaps = compute_to(topic_terms, n=eval_words, return_overlaps=True)
            n_overlaps = np.sum(overlaps == eval_words)

            curr_metrics =  {
                "val_loss": val_loss,
                "npmi": np.mean(npmi_scorer.compute_npmi(topics=topic_terms, n=eval_words)),
                "tu": np.mean(compute_tu(topic_terms, n=eval_words)),
                "to": to,
                "complete_overlaps": n_overlaps,
                "keep": n_overlaps <= max_acceptable_overlap,
            }

            if curr_metrics["keep"]:
                val_metrics['val_loss'] = min(curr_metrics['val_loss'], val_metrics['val_loss'])
                val_metrics['npmi'] = max(curr_metrics['npmi'], val_metrics['npmi'])
                val_metrics['tu'] = max(curr_metrics['tu'], val_metrics['tu'])
                val_metrics['to'] = min(curr_metrics['to'], val_metrics['to'])

            if target_metric is not None and val_metrics[target_metric] == curr_metrics[target_metric]:
                pyro.get_param_store().save(Path(output_dir, "model.pt"))
                save_topics(topic_terms, inv_vocab, Path(output_dir, "topics.txt"), n=topic_words_to_save)

            result_message.update({k: v for k, v in curr_metrics.items()})
            curr_metrics = pd.DataFrame(curr_metrics, index=[epoch])
            curr_metrics.to_csv(
                results_path,
                mode="w" if epoch == 0 else "a",
                header=epoch == 0,
            )
            if save_all_topics:
                save_topics(topic_terms, inv_vocab, Path(output_dir, f"topics/{epoch}.txt"))

        t.set_postfix(result_message)

    plt.plot(epoch_num, loss)
    plt.show()

    if results_path.exists():
        results = pd.read_csv(Path(output_dir, "results.csv"))
        results = results.loc[results.keep]
        print(
            f"Best NPMI: {results.npmi.max():0.4f} @ {np.argmax(results.npmi)}\n"
            f"Best TU @ this NPMI: {results.tu[np.argmax(results.npmi)]:0.4f}\n"
            f"Best TO @ this NPMI: {results.to[np.argmax(results.npmi)]:0.4f}"
        )



    # if target metric is not specified, then save the model at the end 
    if target_metric is None:

        topic_terms = np.flip(vae.decoder.beta.argsort(-1), -1)[:, :topic_words_to_save]
        pyro.get_param_store().save(Path(output_dir, "model.pt"))
        save_topics(topic_terms, inv_vocab, Path(output_dir, "topics.txt"), n=topic_words_to_save)
        # print(topic_terms)
        # print(inv_vocab)

    return vae, results


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message=".*was not registered in the param store because requires_grad=False.*")
    warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

    app()