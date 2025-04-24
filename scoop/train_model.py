"""
Description
===========

Custom training loop for the profiling of a DL-SCA model using Scoop as an optimizer.
Coming from:

    *Scoop: An Optimizer for Profiling Attacks against Higher-Order Masking.*

.. warning::
   Will not work with other optimizers. Comment out Scoop specific lines if you want to use other optimizers.

"""


import torch
import torch.nn.functional as F
import math
import time
import numpy as np

def train_model(model, optimizer, n_epochs, train_loader, valid_loader, hessian_update=10, verbose=False, save_best_model=True, path='best_model.pt', device=None, finetuning=False, entropy=8, MLP=False):
    """
    .. function:: train_model(model, optimizer, n_epochs, train_loader, valid_loader, hessian_update=10, verbose=False, save_best_model=True, path='best_model.pt', device=None, finetuning=False, entropy=8, MLP=False)

   Train a model using a custom training loop that integrates Hessian updates (Scoop-specific).

   :param model: The model to train.
   :param optimizer: The optimizer to use.
   :param n_epochs: The number of epochs for training.
   :param train_loader: Data loader for the training dataset.
   :param valid_loader: Data loader for the validation dataset.
   :param hessian_update: Frequency (in iterations) at which to perform the Hessian update (default is 10).
   :param verbose: If True, prints training progress information (default is False).
   :param save_best_model: If True, saves the best performing model based on validation loss (default is True).
   :param path: File path to save the best model (default is 'best_model.pt').
   :param device: Device on which to run the model (e.g., CPU or GPU; default is None).
   :param finetuning: If True, engages finetuning mode which may alter the stopping criterion (default is False).
   :param entropy: Threshold entropy value used in finetuning mode for stopping the training (default is 8).
   :param MLP: If True, indicates the model is a Multi-Layer Perceptron and modifies input handling accordingly (default is False).

   :returns: A tuple containing:
      - **train_losses**: List of training losses over epochs.
      - **valid_losses**: List of validation losses over epochs.
      - **path**: The file path where the best model was saved.
   :rtype: tuple(list, list, str)

   .. note::
      This training loop includes a Scoop-specific Hessian update line and may not work correctly with other optimizers.
    """
    train_losses = []
    valid_losses = []
    best_val = torch.inf


    start_time = time.time()
    epoch = 0

    n = len(train_loader)

    while True:
        iter = -1
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            if device is not None:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            if not MLP:
                Y_pred = model(X_batch.unsqueeze(1))
            else:
                Y_pred = model(X_batch)
            loss = F.nll_loss(Y_pred, Y_batch)/math.log(2)
            loss.backward(create_graph=True)
            if iter % hessian_update == hessian_update - 1:
                optimizer.hutchinson_hessian() # SCOOP SPECIFIC LINE
            optimizer.step()
            train_loss += loss.item()
            iter += 1
            if verbose:
                print('Iteration ', iter, '/', n, '   Train Loss: ', train_loss/(iter+1), end='\r')
        train_losses.append(train_loss / len(train_loader))
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in valid_loader:
                if device is not None:
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                if not MLP:
                    Y_pred = model(X_batch.unsqueeze(1))
                else:
                    Y_pred = model(X_batch)
                loss = F.nll_loss(Y_pred, Y_batch)/math.log(2)
                valid_loss += loss.item()
        valid_losses.append(valid_loss / len(valid_loader))
        if verbose:
            print(f'Epoch {epoch + 1}/{n_epochs} | '
                  f'Train loss: {train_losses[-1]:.4f} | '
                  f'Valid loss: {valid_losses[-1]:.4f} | '
                  'Expected time left: {:.2f} s'.format((n_epochs - epoch - 1) * (time.time() - start_time) / (epoch + 1)), end='\r')
        if valid_losses[-1] < best_val and save_best_model:
            best_val = valid_losses[-1]
            torch.save(model, path)
        epoch += 1
        np.save('train_losses.npy', train_losses)
        np.save('valid_losses.npy', valid_losses)
        if epoch >= n_epochs:
            if not finetuning:
                break
            else:
                if valid_losses[-1] > entropy:
                    break
        if finetuning and valid_losses[-1] > entropy+0.1:
            break
        if epoch > 1 and math.isnan(valid_losses[-1]):
            break
    return train_losses, valid_losses, path