# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:20:47 2022

@author: arturxe
"""
import torch
import torch.nn.functional as F
from torch import nn

class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, labels, output):
        return torch.mean(torch.mean(labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output)))
    
class NLLLoss_weights(torch.nn.Module):
    def __init__(self):
        super(NLLLoss_weights, self).__init__()
        #self.weights1 = weights1
        
    def forward(self, labels, output):
        weights = torch.tensor([1, 20, 10, 10, 15, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 30, 30]).cuda() #loss online
        #weights = torch.tensor([1, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20]).cuda() #loss ram
        #weights = torch.tensor([1, 40, 20, 30, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 80, 80]).cuda()
        return torch.mean(torch.mean(weights * labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output), dim=0))
    
class MaskLoss(torch.nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, freal, fpreds):
        lossV = (2 - 2 * self.cos(freal, fpreds)).mean()
        return lossV

class CLIP_loss(torch.nn.Module):
    def __init__(self, aux_queues = True):
        super(CLIP_loss, self).__init__()
        self.aux_queues = aux_queues
        #self.weights1 = weights1
        
    def forward(self, classV, classA, aux_negV = None, aux_negA = None):
        batch_size, d = classV.shape
        classV_aux = classV.expand(batch_size, batch_size, d)
        classA_aux = classA.expand(batch_size, batch_size, d)
        negV = classV_aux.flatten()[d:].view(batch_size-1, batch_size+1, d)[:, :-1, :].reshape(batch_size, batch_size-1, d)
        negA = classA_aux.flatten()[d:].view(batch_size-1, batch_size+1, d)[:, :-1, :].reshape(batch_size, batch_size-1, d)

        if aux_negV != None:
            negV = torch.cat((negV, aux_negV.expand(batch_size, aux_negV.shape[0], aux_negV.shape[1])), dim=1)

        if aux_negA != None:
            negA = torch.cat((negA, aux_negA.expand(batch_size, aux_negA.shape[0], aux_negA.shape[1])), dim=1)
        
        loss = InfoNCE(negative_mode='paired')
        
        query = classV
        positive_key = classA
        
        #neg_samplesV = classV.reapeat()
        #negative_keys = 
        return (loss(query, positive_key, negA) + loss(positive_key, query, negV)) / 2



__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]