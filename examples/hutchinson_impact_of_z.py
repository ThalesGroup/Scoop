'''

Description
===========

This script investigates the influence of the sampling distribution's variance and kurtosis on the accuracy of Hutchinson's diagonal estimator. It compares the performance of various distributions (Gaussian, Rademacher, Laplace, Uniform, and Biased Rademacher) both empirically and theoretically.

Key Features
============

1. **Empirical Analysis**:
   - Simulates Hessian matrices.
   - Evaluates error convergence across multiple sampling distributions.

2. **Theoretical Analysis**:
   - Examines the relationship between variance, kurtosis, and expected error.
   - Provides insights into optimal sampling strategies.

Additional Information
======================

This code is part of the paper:

    *Scoop: An Optimizer for Profiling Attacks against Higher-Order Masking*

'''



import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

seed = 42
random.seed(seed)
np.random.seed(seed)


n = 8
n_tries = 100


iterations = 200
z_gauss_convergence = np.zeros(iterations)
z_rademacher_convergence = np.zeros(iterations)
z_laplace_convergence = np.zeros(iterations)
z_uniform_convergence = np.zeros(iterations)
z_biased_rademacher_convergence = np.zeros(iterations)

for _ in tqdm(range(n_tries), desc='Tries'):
    # get hessian_matrix diagonal
    hessian_matrix = np.random.uniform(low=-10, high=10, size=(n,n))

    hessian_matrix_diag = np.diag(hessian_matrix)

    z_gauss_history = np.zeros((len(hessian_matrix_diag), iterations))
    z_rademacher_history = np.zeros((len(hessian_matrix_diag), iterations))
    z_laplace_history = np.zeros((len(hessian_matrix_diag), iterations))
    z_uniform_history = np.zeros((len(hessian_matrix_diag), iterations))
    z_biased_rademacher_history = np.zeros((len(hessian_matrix_diag), iterations))

    for i in tqdm(range(iterations), desc='Iterations', disable=True):
        z_gauss = np.random.normal(0, 1, len(hessian_matrix_diag))
        z_gauss_history[:,i] = z_gauss*(hessian_matrix@z_gauss)
        z_rademacher = np.random.choice([-1,1], len(hessian_matrix_diag))
        z_rademacher_history[:,i] = z_rademacher*(hessian_matrix@z_rademacher)
        z_laplace = np.random.laplace(0, np.sqrt(1/2), len(hessian_matrix_diag))
        z_laplace_history[:,i] = z_laplace*(hessian_matrix@z_laplace)
        z_uniform = np.random.uniform(-np.sqrt(3), np.sqrt(3), len(hessian_matrix_diag))
        z_uniform_history[:,i] = z_uniform*(hessian_matrix@z_uniform) 
        z_biased_rademacher = np.random.choice([-0.9,0.9], len(hessian_matrix_diag))
        z_biased_rademacher_history[:,i] = z_biased_rademacher*(hessian_matrix@z_biased_rademacher)
        clipping = False
        if clipping:
            z_gauss_history[:,i] = np.clip(z_gauss_history[:,i], 0, None)
            z_rademacher_history[:,i] = np.clip(z_rademacher_history[:,i], 0, None)
            z_laplace_history[:,i] = np.clip(z_laplace_history[:,i], 0, None)
            z_uniform_history[:,i] = np.clip(z_uniform_history[:,i], 0, None)
            z_biased_rademacher_history[:,i] = np.clip(z_biased_rademacher_history[:,i], 0, None)



    for i in tqdm(range(iterations), desc='Iterations', disable=True):
        z_gauss_convergence[i] += np.linalg.norm(np.mean(z_gauss_history[:,:i], axis=1)-hessian_matrix_diag,2)/np.linalg.norm(hessian_matrix_diag,2)
        z_rademacher_convergence[i] += np.linalg.norm(np.mean(z_rademacher_history[:,:i], axis=1)-hessian_matrix_diag,2)/np.linalg.norm(hessian_matrix_diag,2)
        z_laplace_convergence[i] += np.linalg.norm(np.mean(z_laplace_history[:,:i], axis=1)-hessian_matrix_diag,2)/np.linalg.norm(hessian_matrix_diag,2)
        z_uniform_convergence[i] += np.linalg.norm(np.mean(z_uniform_history[:,:i], axis=1)-hessian_matrix_diag,2)/np.linalg.norm(hessian_matrix_diag,2)
        z_biased_rademacher_convergence[i] += np.linalg.norm(np.mean(z_biased_rademacher_history[:,:i], axis=1)-hessian_matrix_diag,2)/np.linalg.norm(hessian_matrix_diag,2)

z_gauss_convergence /= n_tries
z_rademacher_convergence /= n_tries
z_laplace_convergence /= n_tries
z_uniform_convergence /= n_tries
z_biased_rademacher_convergence /= n_tries

plt.figure(figsize=(10,6))
plt.plot(z_gauss_convergence, label='Gaussian', color='black', linestyle='--')
plt.plot(z_rademacher_convergence, label='Rademacher', color='black', linestyle='-.')
plt.plot(z_laplace_convergence, label='Laplace', color='black', linestyle=':')
plt.plot(z_uniform_convergence, label='Uniform', color='black', linestyle='-')
plt.plot(z_biased_rademacher_convergence, label='Biased Rademacher', color='red', linestyle='-')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Relative error')
plt.legend()
plt.grid(linestyle='--')
plt.title('Empirical Error computed on Simulated Hessians')
plt.show(block=False)

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


n = 8
m = 10



H = np.random.uniform(low=-10, high=10, size=(n,n))

H_diag = np.diag(H)
H_diag_norm = np.linalg.norm(H_diag, 2)**2
H_bar = np.copy(H)
np.fill_diagonal(H_bar, 0)
H_bar_norm = np.linalg.norm(H_bar, 'fro')**2


theoretical_best_var = (((m-1)/m)*H_diag_norm)/((1/m)*H_bar_norm+H_diag_norm)

print('Diagonal norm:', H_diag_norm)
print('Off-diagonal norm:', H_bar_norm)

print('Theoretical best var:', theoretical_best_var)

var_z = 1
kurt_z = -2

def error_hessian(H, var_z, kurt_z, m=1):
    H_diag = np.diag(H)
    H_diag_norm = np.linalg.norm(H_diag, 2)**2
    H_bar = np.copy(H)
    np.fill_diagonal(H_bar, 0)
    H_bar_norm = np.linalg.norm(H_bar, 'fro')**2
    # return (H_diag_norm - 2*var_z*H_diag_norm + kurt_z*H_diag_norm + var_z**2*H_bar_norm)/m
    variance_error = ((1-2*var_z + kurt_z)*H_diag_norm + var_z**2*H_bar_norm)
    bias_error = (1-var_z)**2*H_diag_norm
    error = variance_error/m + bias_error
    return error/H_diag_norm

def error_convergence(H, var_z, kurt_z, m_max=200):
    m = np.arange(1, m_max+1)
    errors = error_hessian(H, var_z, kurt_z, m)
    return errors

var_z = np.arange(0.1, 2, 0.001)



errors_convergence = error_convergence(H, 1, kurt_z)
errors_convergence_biased = error_convergence(H, 1*(0.9**2), kurt_z*(0.9**3))
errors_convergence_gaussian = error_convergence(H, 1, 0)

best_var_evol = [m*H_diag_norm/H_bar_norm for m in np.arange(1, 101)]

tries = 50
error = np.zeros(len(var_z))
best_var = 0
m=50
for _ in tqdm(range(tries), desc='Computing errors', disable=True):
    H = np.random.uniform(low=-10, high=10, size=(n,n))

    H_diag = np.diag(H)
    H_diag_norm = np.linalg.norm(H_diag, 2)**2
    H_bar = np.copy(H)
    np.fill_diagonal(H_bar, 0)
    H_bar_norm = np.linalg.norm(H_bar, 'fro')**2


    theoretical_best_var = (((m-1)/m)*H_diag_norm)/((1/m)*H_bar_norm+H_diag_norm)
    best_var += theoretical_best_var
    errors = [error_hessian(H, var, kurt_z, m=m) for var in var_z]

    error += errors

errors = error/tries
best_var /= tries

plt.figure(figsize=(10,6))
plt.semilogy(np.arange(1, len(errors_convergence)+1), errors_convergence, label='Unbiased Rademacher', color='black', linestyle='-.')
plt.semilogy(np.arange(1, len(errors_convergence_biased)+1), errors_convergence_biased, label='Biased Rademacher', color='red', linestyle='-')
plt.semilogy(np.arange(1, len(errors_convergence_gaussian)+1), errors_convergence_gaussian, label='Gaussian', color='black', linestyle='-')
plt.grid(linestyle='--')
plt.xlabel('Iterations')
plt.ylabel('Theoretical Expected Error')       
plt.legend()          
plt.title('Theoretical Error as a function of the number of iterations')            
plt.show(block=False)

plt.figure(figsize=(10,6))
plt.semilogy(var_z, errors, label='Error', color='black')
plt.axvline(x=best_var, color='black', linestyle='--', label='Theoretical best var')
plt.grid(linestyle='--')
plt.xlabel('Variance')
plt.ylabel('Theoretical Expected Error')
plt.legend()     
plt.title('Theoretical Error as a function of the variance')
plt.show()


