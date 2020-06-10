import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required


class SGLD(Optimizer):
    """
    Barely modified version of pytorch SGD to implement SGLD
    """

    def __init__(self, params, lr=required, addnoise=True):
        defaults = dict(lr=lr, addnoise=addnoise)
        super(SGLD, self).__init__(params, defaults)

    def step(self, lr=None, add_noise=False):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['addnoise']:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size),
                        torch.ones(size) / np.sqrt(group['lr'])
                    )
                    p.data.add_(-group['lr'],
                                d_p + langevin_noise.sample().cuda())
                else:
                    p.data.add_(-group['lr'], d_p)

        return loss


class SGLDV2(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler
    """

    def __init__(self,
                 params,
                 lr: np.float64 = 1e-2,
                 scale_grad: np.float64 = 1):
        """ Set up a SGLD Optimizer.
        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # if lr_decay is None:
        #     self.lr_decay = const
        #     pass
        # elif lr_decay == "inv":
        #     final_lr_fraction = 1e-2
        #     degree = 2
        #     gamma = (np.power(1 / final_lr_fraction, 1. / degree) - 1) / (T - 1)
        #     self.lr_decay = lambda t: lr * np.power((1 + gamma * t), -degree)
        # else:
        #     self.lr_decay = lr_decay
        defaults = dict(
            lr=lr,
            scale_grad=scale_grad
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr, scale_grad = group["lr"], group["scale_grad"]
                # the average gradient over the batch, i.e N/n sum_i g_theta_i + g_prior
                gradient = parameter.grad.data * scale_grad
                gradient.add_(parameter.data)
                #  State initialization
                if len(state) == 0:
                    state["iteration"] = 0

                sigma = torch.sqrt(torch.from_numpy(
                    np.array(lr, dtype=type(lr))))
                delta = (0.5 * lr * gradient +
                         sigma * torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)))

                parameter.data.add_(-delta)
                state["iteration"] += 1
                state["sigma"] = sigma

        return loss


class pSGLD(Optimizer):
    """
    Barely modified version of pytorch SGD to implement pSGLD
    The RMSprop preconditioning code is mostly from pytorch rmsprop implementation.
    """

    def __init__(self, params, lr=required, alpha=0.99, eps=1e-8, centered=False, addnoise=True):
        defaults = dict(lr=lr, alpha=alpha, eps=eps,
                        centered=centered, addnoise=addnoise)
        super(pSGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(pSGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, lr=None, add_noise=False):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                square_avg.mul_(alpha).addcmul_(1-alpha, d_p, d_p)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1-alpha, d_p)
                    avg = square_avg.cmul(-1, grad_avg,
                                          grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['addnoise']:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size).cuda(),
                        torch.ones(size).cuda().div_(
                            group['lr']).div_(avg).sqrt()
                    )
                    p.data.add_(-group['lr'],
                                d_p.div_(avg) + langevin_noise.sample())
                else:
                    #p.data.add_(-group['lr'], d_p.div_(avg))
                    p.data.addcdiv_(-group['lr'], d_p, avg)

        return loss


class pSGLDV2(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics with noise rescaled in each dimension
        according to RMSProp.
        
        (from https://github.com/automl/pybnn)
    """

    def __init__(self,
                 params,
                 lr=np.float64(1e-2),
                 num_train_points=1,
                 precondition_decay_rate=np.float64(0.99),
                 diagonal_bias=np.float64(1e-5)):
        """ Set up a SGLD Optimizer.
        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        precondition_decay_rate : float, optional
            Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
            Default: `0.99`
        diagonal_bias : float, optional
            Term added to the diagonal of the preconditioner to prevent it from
            degenerating.
            Default: `1e-5`.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(
            lr=lr, precondition_decay_rate=precondition_decay_rate,
            diagonal_bias=diagonal_bias,
            num_train_points=num_train_points
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                num_train_points = group["num_train_points"]
                # alpha
                precondition_decay_rate = group["precondition_decay_rate"]
                diagonal_bias = group["diagonal_bias"]  # lambda
                gradient = parameter.grad.data * num_train_points

                #  state initialization
                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                state["iteration"] += 1

                #  momentum update
                momentum = state["momentum"]
                momentum_t = momentum * precondition_decay_rate + \
                    (1.0 - precondition_decay_rate) * (gradient ** 2)
                state["momentum"] = momentum_t  # V(theta_t+1)

                # compute preconditioner
                preconditioner = (
                    1. / (torch.sqrt(momentum_t) + diagonal_bias))  # G(theta_t+1)

                # standard deviation of the injected noise
                sigma = torch.sqrt(torch.from_numpy(
                    np.array(lr, dtype=type(lr)))) * torch.sqrt(preconditioner)

                mean = 0.5 * lr * (preconditioner * gradient)
                delta = (mean + torch.normal(mean=torch.zeros_like(gradient),
                                             std=torch.ones_like(gradient)) * sigma)

                parameter.data.add_(-delta)

        return loss
