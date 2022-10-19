import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

import numpy as np

import pyro
import pyro.distributions as dist
from pyro.infer import TraceMeanField_ELBO

from utils import log_standard_categorical


# from dvae import LinearSoftmax, CollapsedMultinomial

# model DVAE(
#   (encoder): Encoder(
#     (embedding_layer): Linear(in_features=19973, out_features=100, bias=True)
#     (embed_drop): Dropout(p=0.25, inplace=False)
#     (alpha_layer): Linear(in_features=100, out_features=200, bias=True)
#     (alpha_bn_layer): BatchNorm1d(200, eps=0.001, momentum=0.001, affine=True, track_running_stats=True)
#   )
#   (decoder): Decoder(
#     (eta_layer): Linear(in_features=200, out_features=19973, bias=True)
#     (eta_bn_layer): BatchNorm1d(19973, eps=0.001, momentum=0.001, affine=True, track_running_stats=True)
#   )
# )
class L1RegularizedTraceMeanField_ELBO(TraceMeanField_ELBO):
    def __init__(self, *args, l1_params=None, l1_weight=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.l1_params = l1_params
        self.l1_weight = l1_weight

    @staticmethod
    def l1_regularize(param_names, weight):
        params = torch.cat([pyro.param(p).view(-1) for p in param_names])
        return weight * torch.norm(params, 1)


    def loss_and_grads(self, model, guide, *args, **kwargs):
        loss_standard = self.differentiable_loss(model, guide, *args, **kwargs)
        loss = loss_standard + self.l1_regularize(self.l1_params, self.l1_weight)

        loss.backward()
        loss = loss.item()

        pyro.util.warn_if_nan(loss, "loss")
        return loss


class CollapsedMultinomial(dist.Multinomial):
    """
    Equivalent to n separate `MultinomialProbs(probs, 1)`, where `self.log_prob` treats each
    element of `value` as an independent one-hot draw (instead of `MultinomialProbs(probs, n)`)
    """
    def log_prob(self, value: torch.tensor) -> torch.tensor:
        return ((self.probs + 1e-10).log() * value).sum(-1)


class LinearSoftmax(nn.Linear):
    """
    Linear layer where the weights are first put through a softmax
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO: should we even allow a bias?
        return F.linear(input, F.softmax(self.weight, dim=0), self.bias)

class Encoder(nn.Module):
    def __init__(self,
            vocab_size: int,
            num_topics: int,
            embeddings_dim: int,
            hidden_dim: int,
            dropout: float,):
        super(Encoder, self).__init__()
        # setup linear transformations
        self.embedding_layer = nn.Linear(vocab_size, embeddings_dim)
        self.embed_drop = nn.Dropout(dropout)

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
        # if self.second_hidden_layer:
        #     hidden = F.relu(self.fc(embedded_do))
        #     hidden_do = self.fc_drop(hidden)

        alpha = self.alpha_layer(hidden_do)
        alpha_bn = self.alpha_bn_layer(alpha)

        alpha_pos = torch.max(
            F.softplus(alpha_bn),
            torch.tensor(0.00001, device=alpha_bn.device)
        )
        return alpha_pos


class Decoder(nn.Module):
    def __init__( self,
        vocab_size: int,
        num_topics: int,
        bias_term: bool = True,
        softmax_beta: bool = False,
        beta_init: torch.tensor = None,):
        super(Decoder, self).__init__()
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

class VariationalAutoencoder(nn.Module):
    def __init__(self,
        vocab_size: int,
        num_topics: int,
        alpha_prior: float,
        embeddings_dim: int,
        hidden_dim: int,
        dropout: float,
        bias_term: bool = True,
        softmax_beta: bool = False,
        beta_init: torch.tensor = None,
        cuda: bool = False,):

        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(vocab_size=vocab_size,
            num_topics=num_topics,
            embeddings_dim=embeddings_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,)
        self.decoder = Decoder( vocab_size=vocab_size,
            num_topics=num_topics,
            bias_term=bias_term,
            softmax_beta=softmax_beta,
            beta_init=beta_init,)

        if cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = cuda
        self.num_topics = num_topics
        self.alpha_prior = alpha_prior

    def forward(self, x: torch.tensor,
        bn_annealing_factor: float = 1.0,
        kl_annealing_factor: float = 1.0):


        with pyro.plate("data", x.shape[0]):
            print("hello")
            # use the encoder to get the parameters used to define q(z|x)
            z = self.encoder(x)
            print("here in encoder")
            # sample the latent code z
            with pyro.poutine.scale(None, kl_annealing_factor):
                pyro.sample("doc_topics", dist.Dirichlet(z))

        print(self.num_topics)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)

            alpha_0 = torch.ones(
                x.shape[0], self.num_topics, device=x.device
            ) * self.alpha_prior


            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("doc_topics", dist.Dirichlet(alpha_0))
            print("here in decoder")

            # decode the latent code z
            x_recon = self.decoder(z, bn_annealing_factor)
            # score against actual data
            pyro.sample("obs", CollapsedMultinomial(1, probs=x_recon), obs=x)

            return x_recon


class ElboCalculation(object):
    def __init__(self, x, y, model):
        print("x",x)
        print("y",y)
        recon = model(x)
        print("recon", recon.shape)
        likelihood = F.binary_cross_entropy
        likelihood = -likelihood(recon, x)
        print("likelihood", likelihood)
        prior = -log_standard_categorical(y)
        print("prior", prior)

        # elbo = likelihood + prior - next(self.beta) * self.model.kl_divergence
        # L = self.sampler(elbo)










