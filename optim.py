from collections import defaultdict
import warnings
import math
import torch

from torch.optim import Optimizer


class RAdam(Optimizer):
    r"""Implements RAdam optimization algorithm.
    It has been proposed in `On the Variance of the Adaptive Learning
    Rate and Beyond`__.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.RAdam(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ https://arxiv.org/abs/1908.03265
    Note:
        Reference code: https://github.com/LiyuanLucasLiu/RAdam
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if 'betas' in param and (
                    param['betas'][0] != betas[0]
                    or param['betas'][1] != betas[1]
                ):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    msg = (
                        'RAdam does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(
                        p_data_fp32, memory_format=torch.preserve_format
                    )
                    state['exp_avg_sq'] = torch.zeros_like(
                        p_data_fp32, memory_format=torch.preserve_format
                    )
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (
                        1 - beta2_t
                    )
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = (
                            lr
                            * math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)
                            )
                            / (1 - beta1 ** state['step'])
                        )
                    else:
                        step_size = lr / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if weight_decay != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-weight_decay * lr)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size)

                p.data.copy_(p_data_fp32)

        return loss


class Lookahead(Optimizer):
    r"""Implements Lookahead optimization algorithm.
    It has been proposed in `Lookahead Optimizer: k steps forward, 1
    step back`__
    Arguments:
        optimizer: base inner optimizer optimize, like Yogi, DiffGrad or Adam.
        k: number of lookahead steps (default: 5)
        alpha: linear interpolation factor. 1.0 recovers the inner optimizer.
            (default: 5)
    Example:
        >>> import torch_optimizer as optim
        >>> yogi = optim.Yogi(model.parameters(), lr=0.1)
        >>> optimizer = optim.Lookahead(yogi, k=5, alpha=0.5)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ https://arxiv.org/abs/1907.08610
    Note:
        Reference code: https://github.com/alphadl/lookahead.pytorch
    """

    def __init__(
        self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5
    ) -> None:
        if k < 0.0:
            raise ValueError('Invalid number of lookahead steps: {}'.format(k))
        if alpha < 0:
            raise ValueError(
                'Invalid linear interpolation factor: {}'.format(alpha)
            )

        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group['counter'] = 0

    def _update(self, group):
        for fast in group['params']:
            param_state = self.state[fast]
            if 'slow_param' not in param_state:
                param_state['slow_param'] = torch.clone(fast.data).detach()

            slow = param_state['slow_param']
            fast.data.mul_(self.alpha).add_(slow, alpha=1.0 - self.alpha)
            slow.data.copy_(fast)

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = self.optimizer.step(closure=closure)
        for group in self.param_groups:
            if group['counter'] == 0:
                self._update(group)
            group['counter'] = (group['counter'] + 1) % self.k
        return loss

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.
        It contains two entries:
        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        slow_state_dict = super(Lookahead, self).state_dict()
        fast_state_dict = self.optimizer.state_dict()
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'fast_state': fast_state,
            'slow_state': slow_state_dict['state'],
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict) -> None:
        r"""Loads the optimizer state.
        Arguments:
            state_dict: optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],
        }
        fast_state_dict = {
            'state': state_dict['fast_state'],
            'param_groups': state_dict['param_groups'],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def zero_grad(self) -> None:
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        self.optimizer.zero_grad()

    def __repr__(self) -> str:
        base_str = self.optimizer.__repr__()
        format_string = self.__class__.__name__ + ' ('
        format_string += '\n'
        format_string += 'k: {}\n'.format(self.k)
        format_string += 'alpha: {}\n'.format(self.alpha)
        format_string += base_str
        format_string += '\n'
        format_string += ')'
        return format_string
