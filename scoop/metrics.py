"""
Description
===========

Implementation of the guessing entropy metric for side-channel analysis. This file is part of the paper:

    *Scoop: An Optimizer for Profiling Attacks against Higher-Order Masking.*

"""


import numpy as np
from tqdm import tqdm
import torch

def guessing_entropy(model, X_attack, pt_test, sbox, device, true_key, n_bits=8, n_tries=100, n_attack=None, verbose=False, step=1, batch_pred=False):
    '''.. function:: guessing_entropy(model, X_attack, pt_test, sbox, device, true_key, n_bits=8, n_tries=100, n_attack=None, verbose=False, step=1, batch_pred=False)

   Compute the guessing entropy for a side-channel attack.

   :param model: The prediction model.
   :param X_attack: Features for the attack.
   :param pt_test: Test plaintext values.
   :param sbox: Substitution box for mapping XOR results.
   :param device: Computation device (CPU or GPU).
   :param true_key: The correct key value.
   :param n_bits: Number of bits for key hypotheses (default: 8).
   :param n_tries: Number of iterations to average the entropy (default: 100).
   :param n_attack: Maximum number of attack samples; if None, defaults to range(1, 100, step).
   :param verbose: If True, enables progress output (default: False).
   :param step: Step size for attack sample size increments (default: 1).
   :param batch_pred: If True, perform predictions in batches (default: False).

   :returns: A tuple containing:
      - **GE**: Array of guessing entropy values.
      - **n_attacks**: Array of evaluated attack sample sizes.
   :rtype: tuple(numpy.ndarray, numpy.ndarray)
'''
    if n_attack is None:
        n_attacks = np.arange(1, 100, step)
    else:
        n_attacks = np.arange(1, n_attack, step)
    GE = np.zeros(len(n_attacks)) 
    n_idx = 0
    if not batch_pred:
        with torch.no_grad():
            Y_pred = model(torch.Tensor(X_attack).unsqueeze(1).to(device))
    else:
        Y_pred = []
        batch_step = 32
        for i in tqdm(range(0, len(X_attack), batch_step), disable=not verbose):
            with torch.no_grad():
                Y_pred.append(model(torch.Tensor(X_attack[i:i+batch_step]).unsqueeze(1).to(device)))
        Y_pred = torch.cat(Y_pred)
    for n_attack in tqdm(n_attacks, disable=not verbose):
        for _ in range(n_tries):
            idx = np.random.choice(range(len(X_attack)), n_attack, replace=False).astype(int)
            pts_attack = pt_test[idx]
            ll = np.zeros(2**n_bits)
            probas = Y_pred[idx].cpu().numpy()  
            for i in range(n_attack):
                ll += [probas[i, sbox[j^pts_attack[i]]] for j in range(2**n_bits)]
            key_rank = np.where(np.argsort(ll)[::-1] == true_key)[0][0]
            GE[n_idx] += key_rank
        GE[n_idx] /= n_tries+1
        n_idx +=1 
    return GE, n_attacks        