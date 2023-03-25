import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ini_estim.ml.utilities as utils


def jsd_loss(p, q):
    """ Variational Jensen-Shannon Divergence

    Measures the (negative) divergence between samples from 2
    distributions. For mutual information tasks, P and Q correspond
    to the joint and marginal distributions, respectively. That is,
    P ~ p(x,y), and Q ~ p(x)p(y). In actuality, jsd_loss represents
    an inequality, and to properly measure divergence, P and Q
    must first be transformed into single dimensional data using an
    auxiliary mapping function. This mapping function is typically
    a simple neural network that is learned such that it minimizes
    the output of jsd_loss. Once jsd_loss is minimized, then its value
    is close to the actual divergence value.

    Parameters
    ----------
    p
        Samples from distribution P, transformed by an auxiliary mapping
        function to scalar values.
    q
        Samples from distribution Q, transformed by an auxiliary mapping
        function to scalar values.
    """
    log_2 = np.log(2)
    ep = torch.mean(log_2 - F.softplus(-p))
    en = torch.mean(F.softplus(-q) + q - log_2)
    return en - ep


def mine_loss(p, q):
    """ Donsker-Varadhan representation of KL Divergence

    Measures the (negative) divergence between samples from 2
    distributions. For mutual information tasks, P and Q correspond
    to the joint and marginal distributions, respectively. That is,
    P ~ p(x,y), and Q ~ p(x)p(y). In actuality, mine_loss represents
    an inequality, and to properly measure divergence, P and Q
    must first be transformed into single dimensional data using an
    auxiliary mapping function. This mapping function is typically
    a simple neural network that is learned such that it minimizes
    the output of mine_loss. Once mine_loss is minimized, then its value
    is close to the actual divergence value.

    Parameters
    ----------
    p
        Samples from distribution P, transformed by an auxiliary mapping
        function to scalar values.
    q
        Samples from distribution Q, transformed by an auxiliary mapping
        function to scalar values.
    """
    return -(torch.mean(p) - torch.log(torch.mean(torch.exp(q))))


loss_fun_dict = dict(jsd=jsd_loss, mine=mine_loss)


class InfoMaxLoss(nn.Module):
    def __init__(
            self, discriminator_model_instance : nn.Module, loss_type="jsd"
        ):
        """ Module for info max loss 
        
        Parameters
        ----------
        discriminator_model_instance : nn.Module
            Module for determining in-class / out-of-class encoded data. The 
            module should take (encoded, data) as its parameters and output a
            value for each example in the batch.
        loss_type : str (Optional)
            The type of variational loss to use, either "jsd" (default) or 
            "mine".
        """
        super().__init__()
        self.add_module("discriminator", discriminator_model_instance)
        self.loss_fun = loss_fun_dict[loss_type]
    
    def forward(self, y, x, xp):
        p = self.discriminator(y, x)
        q = self.discriminator(y, xp)
        return self.loss_fun(p, q)


class CPCLoss(nn.Module):
    def __init__(self, num_pred, discriminators=None, num_neg=None, measure="nce"):
        super().__init__()
        self.num_pred = num_pred
        self.num_neg = None
        if isinstance(discriminators, list):
            discriminators = nn.ModuleList(discriminators)
        self.discriminators = discriminators
        self.tau = 1.0
        if measure.lower() == "nce":
            self._measure = self._nce
        elif measure.lower() == "jsd":
            self._measure = self._jsd
    
    def forward(self, c, lengths=None):
        num_batch, max_length, num_dims = c.shape
        winsize = max_length - self.num_pred - 1 
        if lengths is not None:
            lengths = lengths - self.num_pred - 1
        
        # generate negative samples
        num_neg = self.num_neg if self.num_neg is not None else c.shape[0]
        cn = utils.random_sample_sequence(c, num_neg, lengths, winsize)
        c0 = c[:, :winsize, :].contiguous()
        q0 = None
        if self.discriminators is None:
            q0 = F.cosine_similarity(
                c0.expand_as(cn), cn, -1)/self.tau
        else:
            c0multi = c0.expand_as(cn).contiguous()
            if not isinstance(self.discriminators, nn.ModuleList):
                q0 = self.discriminators(c0multi, cn)
        
        if q0 is not None and lengths is not None:
            for i, l in enumerate(lengths):
                q0[i, :, l:] = 0

        # step through predictions
        #  - there is probably a more efficient way to do this...
        loss = 0.0
        for k in range(1, self.num_pred + 1):
            ck = c[:, k:winsize+k, :].contiguous()
            if self.discriminators is None:
                p = F.cosine_similarity(c0, ck, -1)/self.tau
                q = q0
            elif isinstance(self.discriminators, nn.ModuleList):
                p = self.discriminators[k-1](c0, ck)
                q = self.discriminators[k-1](c0multi, cn)
            else:
                p = self.discriminators(c0, ck)
                q = q0

            if lengths is not None:
                for i, l in enumerate(lengths):
                    p[i, l:] = 0
                    if q0 is None:
                        q[i, :, l:] = 0
            
            loss += self._measure(p, q, lengths)
        
        return loss / self.num_pred

    def _nce(self, p, q, lengths=None):
        en = torch.sum(torch.logsumexp(torch.cat((p.unsqueeze(1), q), 1), 1))
        ep = torch.sum(p)
        N = p.numel() if lengths is None else torch.sum(lengths)
        return (en - ep) / max(N, 1)

    def _jsd(self, p, q, lengths=None):
        log_2 = math.log(2)
        Np = p.numel() if lengths is None else torch.sum(lengths)
        Nq = q.numel() if lengths is None else torch.sum(lengths)*q.shape[1]
        ep = torch.sum(log_2 - F.softplus(-p)) / max(Np, 1)
        en = torch.sum(F.softplus(-q) + q - log_2) / max(Nq, 1)
        return en - ep


