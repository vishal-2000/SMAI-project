'''
Author: Vishal Reddy Mandadi 

This class can be used like any other optimizer class and will sharply resemble the functionality of PyTorch's ADAM 
for easy plug and play (though the implementation isn't directly inspired from PyTorch's native implementation, we 
try our best to get the same result out of this one with possibly vectorized computations)
'''
import torch
from torch.optim.optimizer import Optimizer, required
import copy
import math

class ADAM(Optimizer):
    '''
    This class implements the well-known ADAM optimizer proposed in this paper - https://arxiv.org/pdf/1412.6980.pdf
    ADAM includes the advantages of both RMSprop and ADAGRAD as it is built on their combination. 
    Includes bias-correction

    Standard values are used by default. You can pass the custom values during initialization

    Arguements:
    params: an iterable (generally a list of torch.Tensors) that will be optimized (do not send dictionaries or sets)
    # lr: learning rate - not present, use lr instead
    lr: step size
    beta1: decay rate in first moment estimation
    beta2: decay rate in 2nd moment estimation
    eps: epsilon value that adds some non-zero noise to the denominator to prevent it from taking 0 value in the initial steps
    '''
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        if lr < 0 or beta1 < 0 or beta1 < 0 or eps < 0:
            raise ValueError("Invalid params: (lr, beta1, beta2 and epsilon are all supposed to be >=0). Given values: {}, {}, {}".format(lr, beta1, beta2, eps))
        defaults = dict(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        super(ADAM, self).__init__(params, defaults)

    # def __setstate__(self, state):
    #     super(ADAM, self).__setstate__(state)

    def step(self, closure=None):
        '''
        Performs optimization and parameter update. 

        Note: Currently we ignore sparse gradients (as they have large number of zeros in it which wastes both 
              memory and processing power)

        Arguements:
        closure: An arguement that specifies closure to allow the optimizer
                 to recompute the model. (Useful in some other algorithms like Conjugate
                 gradient and LBFGS, not for ADAM) 
        '''
        loss = None
        if closure is not None:
            loss = closure()
        #print('Hi')

        for group in self.param_groups:
            # Get the hyperparameters
            beta1 = group['beta1']
            beta2 = group['beta2']
            lr = group['lr']
            eps = group['eps']

            # Now iteratively update each parameter in the current group
            for p in group['params']:
                if p.grad is None:
                    # if p has no gradient, then we simply skip it
                    continue
                grad = p.grad.data
                # print(p.grad)
                if grad.is_sparse:
                    raise RuntimeError('ADAM does not support sparse gradients, consider using SparseADAM instead')
                # print(p.grad)
                state = self.state[p] # current optimizer state


                if len(state) == 0: 
                    # Implies first time access to the state, so set the defaults
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data) # p
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data) # p

                state['step'] += 1

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # L2 penalty (Regularization stuff)
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Main update step
                # 1. Update exp_avg (exp_avg = exp_avg*beta1 + (1-beta1)*grad)
                exp_avg.mul_(beta1).add_(other=grad, alpha=1-beta1) # inplace update 
                # 2. Update exp_avg_sq (exp_avg_sq = exp_avg_sq*beta2 + (1-beta2)*grad^2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1-beta2) # inplace update
                # 3. Computing bias correction terms
                step = state['step']
                bias_correction1 = 1 - (beta1**step)
                bias_correction2 = 1 - (beta2**step)
                # 4. Compute and add small noise epsilon to the denominator to prevent it from becoming non-zero
                #    The denominator includes its corresponding bias_correction_term2
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps) # exp_avg_sq.sqrt().add_(eps) # (exp_avg_sq.sqrt() / math.sqrt(bias_correction1)).add_(eps)
                # 5. Now calculate the effective step size (lr/bias_correction1)
                step_size = lr/bias_correction1
                #print('step_size: {}'.format(step_size))
                # 6. Update the parameter p
                # print("exp_avg: {}\ndenom: {}\nstep_size: {}".format(exp_avg, denom, step_size))
                # print(step_size*(exp_avg/denom))
                # p = p - step_size*(exp_avg/denom)
                # print(p.data)
                # print(p.grad)
                # print('p.data before: {}'.format(p.data))
                p.data.addcdiv_(exp_avg, denom, value=-1*step_size)
                # print('p.data after: {}'.format(p.data))

        return loss