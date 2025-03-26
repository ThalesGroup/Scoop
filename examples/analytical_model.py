'''Script for simplified DL-SCA model analytical manipulations

Description
===========

This script performs symbolic and numerical computations to analyze and visualize loss functions for DL-SCA. It integrates symbolic computation, numerical methods, and gradient-based optimization techniques (Gradient Descent, Second Order Gradient Descent, Mirror Descent, and Second Order Mirror Descent). The loss landscape and gradients are visualized to help in understanding optimization dynamics. This code generates figures **2.4.1** and **3.2.1** of the paper:

    *Scoop: An Optimizer for Profiling Attacks against Higher-Order Masking.*

Features
--------

- **Symbolic Computation**: Represents and simplifies loss functions using `sympy`.
- **Simulation**: Simulates model outputs, traces, and noise.
- **Parallel Computation**: Processes cumulated loss and loss landscape efficiently using all available cores (set `n_jobs` to limit cores).
- **Optimization Algorithms**: Implements various optimization techniques.
- **Visualization**: Visualizes loss landscapes, gradients, and optimization progress.

Usage
-----

Execute this script in a Python environment with the required libraries installed. Modify parameters such as learning rates, discretization steps, and masking schemes to experiment with different settings.
'''


from src.analysis import *
import matplotlib.ticker as ticker


seed = 42
random.seed = seed
np.random.seed(seed)
n_jobs = -1

theta0, L0 = symbols('theta0 L0')
theta1, L1 = symbols('theta1 L1')

f0 = softmax(single_model(theta0, L0))
f1 = softmax(single_model(theta1, L1))
f = log_outputs(combine_any_models([f0, f1]))

print("Model outputs: ")
print_latex(f[0])
print_latex(f[1])

losses = [NLL(f, 0), NLL(f, 1)]
print("Losses: ")
print_latex(simplify(losses[0]))
print_latex(simplify(losses[1]))

# create a grid of theta values
discretization_step = 0.04
start, stop = -2,2
n_theta_values = int((stop-start)/discretization_step)
theta0_values = np.linspace(start, stop, n_theta_values, dtype=np.float32)
theta1_values = np.linspace(start, stop, n_theta_values, dtype=np.float32)

# simulate traces
alpha0 = -1
alpha1 = 1
sigma = 0
n_traces = 2*(2**2**2)
noise_0 = np.random.normal(0, sigma, n_traces)
noise_1 = np.random.normal(0, sigma, n_traces)
s0 = np.random.randint(0,2,n_traces)
s = np.random.randint(0,2, n_traces)
s1 = s0^s 
L0arr = s0*alpha0+noise_0
L1arr = s1*alpha1+noise_1

print("L0: ", L0)
print("L1: ", L1)

# substitute the values of L0 and L1 in the loss functions

def process_cumulated_loss(i, losses, s, L0arr, L1arr):
    loss = losses[s[i]]
    loss = loss.subs(L0, L0arr[i])
    return simplify(loss.subs(L1, L1arr[i]))

with Parallel(n_jobs=n_jobs) as parallel:
    cumulated_loss_list = parallel(delayed(process_cumulated_loss)(i, losses, s, L0arr, L1arr) for i in tqdm(range(n_traces), desc='Computing cumulated loss'))

cumulated_loss = np.sum(cumulated_loss_list) / n_traces

print("Cumulated loss: ", cumulated_loss)

# compute the loss_landscape

loss_landscape = np.zeros((len(theta0_values), len(theta1_values)), dtype=np.float32)

with Parallel(n_jobs=n_jobs, verbose=0) as parallel: 
    loss_landscape = parallel(
        delayed(calculate_loss)(theta0x, theta1y, theta0, theta1, cumulated_loss)
        for i, theta0x in enumerate(tqdm(theta0_values, desc='Computing loss landscape'))
        for j, theta1y in enumerate(theta1_values)
    )

loss_landscape = np.array(loss_landscape).reshape(len(theta0_values), len(theta1_values)).astype(np.float32)
print("Loss landscape: ", loss_landscape)

loss_landscape_grad = grad_norm(h_grad(loss_landscape), v_grad(loss_landscape))


######## OPTIMIZATION PROCESSES ########
#### Gradient Descent ####

loss_gradient = compute_gradients(cumulated_loss, theta0, theta1)
grad_history = []
mirror_history = []

theta0_init = np.random.uniform(-1/np.sqrt(2),1/np.sqrt(2))
theta1_init = np.random.uniform(-1/np.sqrt(2),1/np.sqrt(2))

learning_rate = 0.1
learning_rate_second = learning_rate/2
print("Learning rate: ", learning_rate)
n_iterations = 100

theta0_values_g = np.zeros(n_iterations)
theta1_values_g = np.zeros(n_iterations)

theta0_values_g[0] = theta0_init
theta1_values_g[0] = theta1_init
loss_values_g = np.zeros(n_iterations)
loss_values_g[0] = cumulated_loss.subs(theta0, theta0_values_g[0]).subs(theta1, theta1_values_g[0]).evalf()
theta_values_g = np.array([theta0_values_g, theta1_values_g])


for i in tqdm(range(1, n_iterations), desc='Gradient Descent', disable=True):
    theta_values_g[:,i] = theta_values_g[:,i-1] - learning_rate*np.array([loss_gradient[j].subs(theta0, theta_values_g[0,i-1]).subs(theta1, theta_values_g[1,i-1]).evalf() for j in range(2)])
    loss_values_g[i] = cumulated_loss.subs(theta0, theta_values_g[0,i]).subs(theta1, theta_values_g[1,i]).evalf()
    print("Gradient Descent It: ", i, "/", n_iterations, " | Loss: ", loss_values_g[i], end='\r')
    theta0_values_g[i], theta1_values_g[i] = theta_values_g[0,i], theta_values_g[1,i]


### Second Order Gradient Descent ###

loss_hessian = compute_hessian(cumulated_loss, theta0, theta1)
hessian_history = []

theta0_values_h = np.zeros(n_iterations)
theta1_values_h = np.zeros(n_iterations)

theta0_values_h[0] = theta0_init
theta1_values_h[0] = theta1_init
loss_values_h = np.zeros(n_iterations)

loss_values_h[0] = cumulated_loss.subs(theta0, theta0_values_h[0]).subs(theta1, theta1_values_h[0]).evalf()
theta_values_h = np.array([theta0_values_h, theta1_values_h])

for i in tqdm(range(1, n_iterations), desc='Second Order Gradient Descent', disable=True):
    #print(loss_hessian)
    hessian = clip_diag(hutchinson(list_of_list_to_np(evaluate_hessian(loss_hessian, theta_values_h[0,i-1], theta_values_h[1,i-1], theta0, theta1))), min=1e-15)
    hessian_inv = clip_diag(1/hessian, min=None, max=10)
    theta_values_h[:,i] = theta_values_h[:,i-1] - learning_rate_second*hessian_inv*np.array([loss_gradient[j].subs(theta0, theta_values_h[0,i-1]).subs(theta1, theta_values_h[1,i-1]).evalf() for j in range(2)])
    loss_values_h[i] = cumulated_loss.subs(theta0, theta_values_h[0,i]).subs(theta1, theta_values_h[1,i]).evalf()
    print("Second Order Gradient Descent | It: ", i, "/", n_iterations, " | Loss: ", loss_values_h[i], " | Hessian: ", hessian.flatten(), end='\r')
    theta0_values_h[i], theta1_values_h[i] = theta_values_h[0,i], theta_values_h[1,i]
    

### Mirror Descent ###

theta0_values_m = np.zeros(n_iterations)
theta1_values_m = np.zeros(n_iterations)
theta0_values_m[0] = theta0_init
theta1_values_m[0] = theta1_init
loss_values_m = np.zeros(n_iterations)
loss_values_m[0] = cumulated_loss.subs(theta0, theta0_values_m[0]).subs(theta1, theta1_values_m[0]).evalf()
theta_values_m = np.array([theta0_values_m, theta1_values_m])


eps = 0.1
for i in tqdm(range(1, n_iterations), desc='Mirror Descent', disable=True):
    theta_update = (1+eps)*(np.abs(theta_values_m[:,i-1])**eps)*np.sign(theta_values_m[:,i-1])-learning_rate*np.array([loss_gradient[j].subs(theta0, theta_values_m[0,i-1]).subs(theta1, theta_values_m[1,i-1]).evalf() for j in range(2)])
    theta_values_m[:,i] = (np.abs(theta_update/(1+eps))**(1/eps))*np.sign(theta_update)
    loss_values_m[i] = cumulated_loss.subs(theta0, theta_values_m[0,i]).subs(theta1, theta_values_m[1,i]).evalf()
    print("Miror Descent | It: ", i, "/", n_iterations, " | Loss: ", loss_values_m[i], end='\r')
    theta0_values_m[i], theta1_values_m[i] = theta_values_m[0,i], theta_values_m[1,i]

### Second order mirror descent ###

theta0_values_m2 = np.zeros(n_iterations)
theta1_values_m2 = np.zeros(n_iterations)
theta0_values_m2[0] = theta0_init
theta1_values_m2[0] = theta1_init

loss_values_m2 = np.zeros(n_iterations)
loss_values_m2[0] = cumulated_loss.subs(theta0, theta0_values_m2[0]).subs(theta1, theta1_values_m2[0]).evalf()
theta_values_m2 = np.array([theta0_values_m2, theta1_values_m2])
eps = 0.1
for i in tqdm(range(1, n_iterations), desc='Second Order Mirror Descent', disable=True):
    hessian = clip_diag(hutchinson(list_of_list_to_np(evaluate_hessian(loss_hessian, theta_values_m2[0,i-1], theta_values_m2[1,i-1], theta0, theta1))), min=1e-15)
    hessian_inv = clip_diag(1/hessian, min=None, max=10)
    theta_update = (1+eps)*(np.abs(theta_values_m2[:,i-1])**eps)*np.sign(theta_values_m2[:,i-1])-learning_rate_second*hessian_inv*np.array([loss_gradient[j].subs(theta0, theta_values_m2[0,i-1]).subs(theta1, theta_values_m2[1,i-1]).evalf() for j in range(2)])
    theta_values_m2[:,i] = (np.abs(theta_update/(1+eps))**(1/eps))*np.sign(theta_update)
    loss_values_m2[i] = cumulated_loss.subs(theta0, theta_values_m2[0,i]).subs(theta1, theta_values_m2[1,i]).evalf()
    print("Second Order Miror Descent | It: ", i, "/", n_iterations, " | Loss: ", loss_values_m2[i], end='\r')
    theta0_values_m2[i], theta1_values_m2[i] = theta_values_m2[0,i], theta_values_m2[1,i]

loss_landscape_global_min = np.min(loss_landscape)
theta0_globalmin, theta1_globalmin = np.where(loss_landscape == loss_landscape_global_min)
grad_slice = get_gradient_slice(loss_landscape_grad, theta0_globalmin[0])

loss_landscape = np.transpose(loss_landscape)

diff_loss_g = np.abs(np.diff(loss_values_g))
diff_loss_m = np.abs(np.diff(loss_values_m))
diff_loss_h = np.abs(np.diff(loss_values_h))
diff_loss_m2 = np.abs(np.diff(loss_values_m2))

plt.figure()
plt.plot(loss_values_g, label='Gradient Descent')
plt.plot(loss_values_m, label='Mirror Descent')
plt.plot(loss_values_h, label='Second Order Gradient Descent')
plt.plot(loss_values_m2, label='Second Order Mirror Descent')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(diff_loss_g, label='Gradient Descent')
plt.plot(diff_loss_m, label='Mirror Descent')
plt.plot(diff_loss_h, label='Second Order Gradient Descent')
plt.plot(diff_loss_m2, label='Second Order Mirror Descent')
plt.xlabel('Iterations')
plt.ylabel('delta_t')
plt.legend()
plt.show(block=False)


print(np.shape(loss_landscape))
print(loss_landscape)
xx, yy = np.meshgrid(theta0_values, theta1_values)

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${}e{}$'.format(a, b)

levels = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1, 1.025, 1.05 ,1.15, 1.30, 1.5, 1.8, 2.5, 3.5, 5]

plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots()
contour = ax.contourf(xx, yy, loss_landscape, levels=levels, cmap='plasma', extend='both', vmin=0.5, vmax=1.2)
contour_level = ax.contour(xx, yy, loss_landscape, levels=levels, colors=('k',),  linewidths=(0.5,))
ax.clabel(contour_level, fmt='%2.1f', colors='k', fontsize=9)
cbar = fig.colorbar(contour)
cbar.ax.set_ylabel('Loss')
cbar.add_lines(contour_level)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
plt.show(block=False)

levels_grad = np.linspace(0,0.03, 15)

field_x, field_y = np.gradient(-loss_landscape)
field_x, field_y = field_x / np.sqrt(field_x**2+field_y**2), field_y / np.sqrt(field_x**2+field_y**2)
field_step = 1

field_step = 10
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots()
contour = ax.contourf(xx, yy, loss_landscape_grad, levels=levels_grad, cmap='plasma', extend='both')
contour_level = ax.contour(xx, yy, loss_landscape_grad, levels=levels_grad, colors=('k',),  linewidths=(0.5,))
ax.streamplot(xx[::field_step], yy[::field_step], field_y[::field_step], field_x[::field_step], color='w', linewidth=0.5)
cbar = fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
cbar.ax.set_ylabel('Gradient of the Loss')
cbar.add_lines(contour_level)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
plt.show(block=False)

theta0_values_dual = project_axis(theta0_values, eps=0.1)
theta1_values_dual = project_axis(theta1_values, eps=0.1)
xx, yy = np.meshgrid(theta0_values_dual, theta1_values_dual)


fig, ax = plt.subplots()
contour = ax.contourf(xx, yy, loss_landscape, levels=levels, cmap='plasma', extend='both', vmin=0.5, vmax=1.2)
contour_level = ax.contour(xx, yy, loss_landscape, levels=levels, colors=('k',),  linewidths=(0.5,))
ax.clabel(contour_level, fmt='%2.1f', colors='w', fontsize=11)
cbar = fig.colorbar(contour)
cbar.ax.set_ylabel('Loss')
cbar.add_lines(contour_level)
ax.set_xlabel(r'$\nabla\psi\left(\theta_0\right)$')
ax.set_ylabel(r'$\nabla\psi\left(\theta_1\right)$')
plt.show(block=False)

fig, ax = plt.subplots()
contour = ax.contourf(xx, yy, loss_landscape_grad, levels=levels_grad, cmap='plasma', extend='both')
contour_level = ax.contour(xx, yy, loss_landscape_grad, levels=levels_grad, colors=('k',),  linewidths=(0.5,))
ax.clabel(contour_level, fmt=ticker.FuncFormatter(fmt), colors='w', fontsize=11)
cbar = fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
cbar.ax.set_ylabel('Gradient of the Loss')
cbar.add_lines(contour_level)
ax.set_xlabel(r'$\nabla\psi\left(\theta_0\right)$')
ax.set_ylabel(r'$\nabla\psi\left(\theta_1\right)$')
plt.show()
