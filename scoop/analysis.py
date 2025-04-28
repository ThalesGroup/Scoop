'''
Description
=========================

This file contains a collection of functions and utilities designed for DL-SCA mathematical modeling, gradient computation, and optimization within a specific framework detailed in section 2 of the paper:

    *Scoop: An Optimizer for Profiling Attacks against Higher-Order Masking.*

The code primarily leverage symbolic and numerical computation libraries to define, manipulate, and evaluate loss functions and their derivatives.

Non-Exhaustive List of Dependencies
=====================================

- **sympy**: For symbolic mathematics, including differentiation and simplification.
- **joblib**: For parallelization and task execution. By default, the code uses all available cores for parallel computation. Set `n_jobs` to a specific number to limit the number of cores used.

Key Functionalities
===================

1. **Mathematical Models**  
   Define and combine single-parameter models into more complex schemes.

2. **Gradient and Loss Calculations**  
   Compute gradients, loss values, and their symbolic derivatives for optimization.

3. **Hessian and Estimators**  
   Evaluate and manipulate the Hessian matrix for advanced optimization techniques.

4. **Loss Landscape Analysis**  
   Compute horizontal and vertical gradients and gradient norms of the loss landscape.

5. **Projection and Mapping**  
   Perform parameter projection onto dual spaces.

Citation Notice
===============

If you are using this code, please cite the aforementioned paper in your work.
'''


import numpy as np
from sympy import symbols, log, diff, exp, print_latex, simplify, Abs, sqrt
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def single_model(theta, L):
    '''
    .. function:: single_model(theta, L)

    Single parameter model :math:`\mathbf{F}_i(\alpha_i s_i \mid \theta_i)` as per Eq. 2.3.2.

    :param theta: Parameter of the model
    :param L: SCA Leakage
    :returns: A list of two values:
        - **M0**: Computed as ``theta * L``.
        - **1 - M0**: The complementary value to M0.
    '''
    M0 = theta*L
    f = [M0, 1-M0]
    return f

def combine_any_models(f):
    '''
    .. function:: combine_any_models(f)

    Combine n+1 single parameter models into the scheme-aware model.

    Based on Eq. 2.3.3, this function uses the Poisson equation to compute the convolution and presupposes a boolean masking scheme.

    :param f: A list of single parameter models.
    :returns: A list containing two simplified values corresponding to the scheme-aware model outputs.
    '''
    n_bits = len(f)
    M0 = 0
    M1 = 0
    for i in range(2**n_bits):
        i_bits = [int(x) for x in list(bin(i)[2:].zfill(n_bits))]
        if sum(i_bits) % 2 == 0:
            M0 += np.prod([f[j][i_bits[j]] for j in range(n_bits)])
        else:
            M1 += np.prod([f[j][i_bits[j]] for j in range(n_bits)])
    return [simplify(M0), simplify(M1)]

def softmax(f):
    '''
    .. function:: softmax(f)

    Compute the arg-softmax function for an :math:`n=1` scheme-aware model.

    To generalize to any :math:`n`, loop similarly to :func:`combine_any_models`.

    :param f: A list or tuple with two elements representing the outputs of a scheme-aware model.
    :returns: A list containing two simplified softmax values computed as:

        - :math:`\frac{\exp(f[0])}{\exp(f[0]) + \exp(f[1])}`
        - :math:`\frac{\exp(f[1])}{\exp(f[0]) + \exp(f[1])}`
    '''
    return [simplify(exp(f[0])/(exp(f[0])+exp(f[1]))), simplify(exp(f[1])/(exp(f[0])+exp(f[1])))]    

def log_outputs(f):
    for i in range(len(f)):
        f[i] = simplify(log(f[i]))
    return f

def NLL(f, s):
    '''
    .. function:: NLL(f, s)

    Computes the negative log-likelihood of the model given the secret :math:`s`.

    :param f: A list or tuple containing two numerical values corresponding to the model outputs.
    :param s: The secret value (expected to be 0 or 1) for which the likelihood is computed.
    :returns: The negative log-likelihood computed as :math:`-\frac{f[s]}{\log(2)}`.
    '''
    if s == 0:
        return -f[0]/np.log(2)
    if s == 1:
        return -f[1]/np.log(2)

def compute_gradients(loss, theta0, theta1):
    '''
    .. function:: compute_gradients(loss, theta0, theta1)

    Compute the gradient of the loss function with respect to the parameters :math:`\theta_0` and :math:`\theta_1`.

    :param loss: The loss function.
    :param theta0: The first parameter.
    :param theta1: The second parameter.
    :returns: A list containing the gradients of the loss with respect to :math:`\theta_0` and :math:`\theta_1`.
    '''
    loss_g = [diff(loss, theta0), diff(loss, theta1)]
    return loss_g

def calculate_loss(theta0x, theta1y, theta0, theta1, cumulated_loss):
    '''
    .. function:: calculate_loss(theta0x, theta1y, theta0, theta1, cumulated_loss)

    Evaluate the loss function given the parameters :math:`\theta_0` and :math:`\theta_1`.

    :param theta0x: Numerical value to substitute for :math:`\theta_0`.
    :param theta1y: Numerical value to substitute for :math:`\theta_1`.
    :param theta0: Symbolic variable representing :math:`\theta_0`.
    :param theta1: Symbolic variable representing :math:`\theta_1`.
    :param cumulated_loss: Symbolic expression for the cumulated loss function.
    :returns: The evaluated loss value as a floating point number.
    '''
    return cumulated_loss.subs(theta0, theta0x).subs(theta1, theta1y).evalf()

def l1_eps_mirror_map(x, eps=0.1):
    '''.. function:: l1_eps_mirror_map(x, eps=0.1)

    Compute the L1 epsilon mirror map transformation.

    The transformation is defined as:

    .. math::
        (1+\epsilon)|x|^{\epsilon}\operatorname{sign}(x)

    :param x: Input value(s) to be transformed.
    :param eps: Epsilon parameter used in the transformation (default: 0.1).
    :returns: The transformed value(s).
    '''
    return (1+eps)*(np.abs(x)**eps)*np.sign(x)

def project_axis(thetas, eps=0.1):
    '''
    .. function:: project_axis(thetas, eps=0.1)

    Project the parameters :math:`\theta_0` and :math:`\theta_1` onto the :math:`\ell_{1+\epsilon}` ball.

    :param thetas: A list of parameter values.
    :param eps: Epsilon value for the mirror map transformation (default: 0.1).
    :returns: A list of projected parameter values after applying the L1 epsilon mirror map.
    '''
    thetas_dual = thetas
    for i in range(len(thetas)):
        thetas_dual[i] = l1_eps_mirror_map(thetas[i], eps=eps)
    return thetas_dual

def compute_hessian(loss, theta0, theta1):
    '''
    .. function:: compute_hessian(loss, theta0, theta1)

    Compute the Hessian of the loss function with respect to the parameters :math:`\theta_0` and :math:`\theta_1`.

    :param loss: The loss function as a symbolic expression.
    :param theta0: The symbolic variable representing :math:`\theta_0`.
    :param theta1: The symbolic variable representing :math:`\theta_1`.
    :returns: A 2x2 list representing the Hessian matrix of the loss function.
    '''
    loss_h = [[diff(diff(loss, theta0), theta0), diff(diff(loss, theta0), theta1)],
              [diff(diff(loss, theta1), theta0), diff(diff(loss, theta1), theta1)]]
    return loss_h

def hutchinson(hessian):
    '''
    .. function:: hutchinson(hessian)

    Compute the Hutchinson estimator of the Hessian.

    This function estimates the Hessian by multiplying it with a random vector whose entries are chosen from {-1, 1} and then performing an element-wise product with the same random vector.

    :param hessian: A Hessian matrix (e.g., as a NumPy array) for which the estimator is computed.
    :returns: The Hutchinson estimator of the Hessian.
    '''
    z = np.random.choice([-1,1], 2)
    hessian_vp = hessian@z
    return hessian_vp*z

def clip_diag(hessian, min=1e-7, max = None):
    '''
    .. function:: clip_diag(hessian, min=1e-7, max=None)

    Clip the values of the Hessian matrix using the provided minimum and/or maximum thresholds.

    :param hessian: The Hessian matrix or array to be clipped.
    :param min: The lower bound for clipping. If provided and :math:`max` is None, values below this threshold are clipped (default: 1e-7).
    :param max: The upper bound for clipping. If provided and :math:`min` is None, values above this threshold are clipped.
    :returns: The clipped Hessian matrix.
    '''
    if max is None and min is not None:
        return np.clip(hessian, min, None)
    elif min is None and max is not None:
        return np.clip(hessian, None, max)
    elif min is not None and max is not None:
        return np.clip(hessian, min, max)
    else:
        return hessian

def list_of_list_to_np(hessian):
    '''
    .. function:: list_of_list_to_np(hessian)

    Convert a 2x2 list of lists into a NumPy array.

    :param hessian: A 2x2 list of lists containing numerical values.
    :returns: A 2x2 NumPy array with the same values.
    '''
    hessian_np = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            hessian_np[i,j] = hessian[i][j]
    return hessian_np

def evaluate_hessian(hessian, theta0x, theta1y, theta0, theta1, epsilon=1e-7):
    '''
    .. function:: evaluate_hessian(hessian, theta0x, theta1y, theta0, theta1, epsilon=1e-7)

    Evaluates the Hessian at a given point :math:`(\theta_0, \theta_1)`.

    :param hessian: A 2x2 Hessian matrix represented as a list of lists with symbolic expressions.
    :param theta0x: The numerical value to substitute for :math:`\theta_0`.
    :param theta1y: The numerical value to substitute for :math:`\theta_1`.
    :param theta0: The symbolic variable representing :math:`\theta_0`.
    :param theta1: The symbolic variable representing :math:`\theta_1`.
    :param epsilon: A small value added to each Hessian element for numerical stability (default: 1e-7).
    :returns: The evaluated and adjusted Hessian as a 2x2 list.
    '''
    for i in range(2):
        for j in range(2):
            value = hessian[i][j].subs(theta0, theta0x).subs(theta1, theta1y).evalf()
            value += epsilon
            hessian[i][j] = value
    return hessian

def h_grad(loss_landscape):
    '''
    .. function:: h_grad(loss_landscape)

    Compute the horizontal gradient of the loss landscape.

    This function calculates the gradient along the horizontal axis of the provided loss landscape. It uses a forward difference at the first row, a backward difference at the last row, and a central difference for the intermediate rows.

    :param loss_landscape: A 2D NumPy array representing the loss landscape.
    :returns: A 2D NumPy array of the same shape as ``loss_landscape`` containing the computed horizontal gradient.
    :rtype: numpy.ndarray
    '''
    h_grad = np.zeros_like(loss_landscape)
    for i in range(np.shape(loss_landscape)[0]):
        for j in range(np.shape(loss_landscape)[1]):
            if i == 0:
                h_grad[i,j] = loss_landscape[i+1,j]-loss_landscape[i,j]
            elif i == np.shape(loss_landscape)[0]-1:
                h_grad[i,j] = loss_landscape[i,j]-loss_landscape[i-1,j]
            else:
                h_grad[i,j] = (loss_landscape[i+1,j]-loss_landscape[i-1,j])/2
    return h_grad

def v_grad(loss_landscape):
    '''
    .. function:: v_grad(loss_landscape)

    Compute the vertical gradient of the loss landscape.

    This function calculates the gradient along the vertical axis of the provided loss landscape. It uses a forward difference at the first column, a backward difference at the last column, and a central difference for intermediate columns.

    :param loss_landscape: A 2D NumPy array representing the loss landscape.
    :returns: A 2D NumPy array of the same shape as ``loss_landscape`` containing the computed vertical gradient.
    :rtype: numpy.ndarray
    '''
    v_grad = np.zeros_like(loss_landscape)
    for i in range(np.shape(loss_landscape)[0]):
        for j in range(np.shape(loss_landscape)[1]):
            if j == 0:
                v_grad[i,j] = loss_landscape[i,j+1]-loss_landscape[i,j]
            elif j == np.shape(loss_landscape)[1]-1:
                v_grad[i,j] = loss_landscape[i,j]-loss_landscape[i,j-1]
            else:
                v_grad[i,j] = (loss_landscape[i,j+1]-loss_landscape[i,j-1])/2
    return v_grad

def grad_norm(h_grad, v_grad):
    '''
    .. function:: grad_norm(h_grad, v_grad)

    Compute the norm of the gradient of the loss landscape.

    The gradient norm is computed as the square root of the sum of the squares of the horizontal and vertical gradients.

    :param h_grad: A 2D NumPy array representing the horizontal gradient of the loss landscape.
    :param v_grad: A 2D NumPy array representing the vertical gradient of the loss landscape.
    :returns: A 2D NumPy array containing the computed gradient norm.
    :rtype: numpy.ndarray
    '''
    return np.sqrt(h_grad**2+v_grad**2)

def get_gradient_slice(ll_grad, thetax):
    '''
    .. function:: get_gradient_slice(ll_grad, thetax)

    Takes a slice of the gradient landscape at a given :math:`\theta_x`.

    :param ll_grad: A 2D NumPy array representing the gradient landscape.
    :param thetax: The index (coordinate) along the x-axis at which to extract the gradient slice.
    :returns: A list containing the gradient values at the specified :math:`\theta_x` coordinate.
    :rtype: list
    '''
    n,m = np.shape(ll_grad)
    grad_slice = []
    i = 0
    while i < m:
        grad_slice.append(ll_grad[thetax,i])
        i += 1
    return grad_slice