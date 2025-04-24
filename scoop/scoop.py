"""
PyTorch implementation of the Scoop optimizer (pseudo code is given in [Sec. 5, 1])

Scoop is a second-order stochastic mirror descent algorithm that uses a biased estimator of the Hessian matrix to precondition the gradient. It is designed to be used with large batch sizes and is particularly effective for training deep neural networks.
The mirror map is chosen to be the $\ell_1$ norm to encourage sparsity in the parameters.

This code is a originally a fork from the second-order stochastic optimization algorithm: Sophia [2]

[1] https://eprint.iacr.org/2025/498
[2] https://arxiv.org/pdf/2305.14342
"""



import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List


class Scoop(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho = 0.04,
         weight_decay=0, *, maximize: bool = False, estimator='classic', hessian_iter=5):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho parameter at index 1: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, rho=rho, 
                        weight_decay=weight_decay, 
                        maximize=maximize, estimator=estimator, 
                        hessian_iter=hessian_iter)
        super(Scoop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def hutchinson_hessian(self):
        '''
        Biased estimator of the Hessian matrix using a scaled version of the Hutchinson estimator. (Section 3.3.2, Thm. 3)
        '''
        for group in self.param_groups:
            _, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if False else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                def grad_fn(x):
                    return p.grad
                v = torch.randint(0, 2, p.size(), device=p.device).float()
                v[v==0] = -1
                if group['estimator'] == 'classic':
                    Hv = torch.func.jvp(grad_fn, (p,), (v,))[1]
                    state['hessian'].addcmul_(v, Hv, value=1)
                elif group['estimator'] == 'low_variance':
                    vHv = torch.zeros_like(p, memory_format=torch.preserve_format)
                    v = 9*v/10
                    for _ in range(group['hessian_iter']):
                        vHv.addcmul_(v, torch.func.jvp(grad_fn, (p,), (v,))[1], value=1)
                    state['hessian'].add_(vHv/group['hessian_iter'])
                    state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if False else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)                
                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])

            scoop(params_with_grad,
                  grads,
                  exp_avgs,
                  hessian,
                  state_steps,
                  bs=bs,
                  beta1=beta1,
                  beta2=beta2,
                  rho=group['rho'],
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  maximize=group['maximize'])

        return loss

def scoop(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          hessian: List[Tensor],
          state_steps: List[Tensor],
          *,
          bs: int,
          beta1: float,
          beta2: float,
          rho: float,
          lr: float,
          weight_decay: float,
          maximize: bool):

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")
    
    func = _single_tensor_scoop
    func(params,
         grads,
         exp_avgs,
         hessian,
         state_steps,
         bs=bs,
         beta1=beta1,
         beta2=beta2,
         rho=rho,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize)

def _single_tensor_scoop(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         hessian: List[Tensor],
                         state_steps: List[Tensor],
                         *,
                         bs: int,
                         beta1: float,
                         beta2: float,
                         rho: float,
                         lr: float,
                         weight_decay: float,
                         maximize: bool):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]            
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)
        step_t += 1
        param.mul_(1 - lr * weight_decay)   # + lambda*||w||_2^2
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        step_size_neg = - lr 
        ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None,1)
        eps = 0.1
        param.copy_(
            (
                (
                    param.abs().pow(eps).mul(param.sign())
                    .add(exp_avg.sign().mul(step_size_neg * ratio))
                ).abs().pow(1 / eps)
            ).mul(
                (
                    param.abs().pow(eps).mul(param.sign())
                    .add(exp_avg.sign().mul(step_size_neg * ratio))
                ).sign()
            )
        )