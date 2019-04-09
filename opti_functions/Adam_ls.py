import math
import torch
from torch.optim import Optimizer
import numpy as np
from copy import deepcopy
from functools import reduce

def is_legal(v):
    """
    Checks that tensor is not NaN or Inf.

    Inputs:
        v (tensor): tensor to be checked

    """
    legal = not torch.isnan(v).any() and not torch.isinf(v)

    return legal

def polyinterp(points, x_min_bound=None, x_max_bound=None, plot=False):
    """
    Gives the minimizer and minimum of the interpolating polynomial over given points
    based on function and derivative information. Defaults to bisection if no critical
    points are valid.

    Based on polyinterp.m Matlab function in minFunc by Mark Schmidt with some slight
    modifications.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 12/6/18.

    Inputs:
        points (nparray): two-dimensional array with each point of form [x f g]
        x_min_bound (float): minimum value that brackets minimum (default: minimum of points)
        x_max_bound (float): maximum value that brackets minimum (default: maximum of points)
        plot (bool): plot interpolating polynomial

    Outputs:
        x_sol (float): minimizer of interpolating polynomial
        F_min (float): minimum of interpolating polynomial

    Note:
      . Set f or g to np.nan if they are unknown

    """
    no_points = points.shape[0]
    order = np.sum(1 - np.isnan(points[:,1:3]).astype('int')) - 1

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])

    # compute bounds of interpolation area
    if(x_min_bound is None):
        x_min_bound = x_min
    if(x_max_bound is None):
        x_max_bound = x_max

    # explicit formula for quadratic interpolation
    if no_points == 2 and order == 2 and plot is False:
        # Solution to quadratic interpolation is given by:
        # a = -(f1 - f2 - g1(x1 - x2))/(x1 - x2)^2
        # x_min = x1 - g1/(2a)
        # if x1 = 0, then is given by:
        # x_min = - (g1*x2^2)/(2(f2 - f1 - g1*x2))

        if(points[0, 0] == 0):
            x_sol = -points[0, 2]*points[1, 0]**2/(2*(points[1, 1] - points[0, 1] - points[0, 2]*points[1, 0]))
        else:
            a = -(points[0, 1] - points[1, 1] - points[0, 2]*(points[0, 0] - points[1, 0]))/(points[0, 0] - points[1, 0])**2
            x_sol = points[0, 0] - points[0, 2]/(2*a)

        x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)

    # explicit formula for cubic interpolation
    elif no_points == 2 and order == 3 and plot is False:
        # Solution to cubic interpolation is given by:
        # d1 = g1 + g2 - 3((f1 - f2)/(x1 - x2))
        # d2 = sqrt(d1^2 - g1*g2)
        # x_min = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2))
        d1 = points[0, 2] + points[1, 2] - 3*((points[0, 1] - points[1, 1])/(points[0, 0] - points[1, 0]))
        d2 = np.sqrt(d1**2 - points[0, 2]*points[1, 2])
        if np.isreal(d2):
            x_sol = points[1, 0] - (points[1, 0] - points[0, 0])*((points[1, 2] + d2 - d1)/(points[1, 2] - points[0, 2] + 2*d2))
            x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)
        else:
            x_sol = (x_max_bound + x_min_bound)/2

    # solve linear system
    else:
        # define linear constraints
        A = np.zeros((0, order+1))
        b = np.zeros((0, 1))

        # add linear constraints on function values
        for i in range(no_points):
            if not np.isnan(points[i, 1]):
                constraint = np.zeros((1, order+1))
                for j in range(order, -1, -1):
                    constraint[0, order - j] = points[i, 0]**j
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 1])

        # add linear constraints on gradient values
        for i in range(no_points):
            if not np.isnan(points[i, 2]):
                constraint = np.zeros((1, order+1))
                for j in range(order):
                    constraint[0, j] = (order-j)*points[i,0]**(order-j-1)
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 2])

        # check if system is solvable
        if(A.shape[0] != A.shape[1] or np.linalg.matrix_rank(A) != A.shape[0]):
            x_sol = (x_min_bound + x_max_bound)/2
            f_min = np.Inf
        else:
            # solve linear system for interpolating polynomial
            coeff = np.linalg.solve(A, b)

            # compute critical points
            dcoeff = np.zeros(order)
            for i in range(len(coeff) - 1):
                dcoeff[i] = coeff[i]*(order-i)

            crit_pts = np.array([x_min_bound, x_max_bound])
            crit_pts = np.append(crit_pts, points[:, 0])

            if not np.isinf(dcoeff).any():
                roots = np.roots(dcoeff)
                crit_pts = np.append(crit_pts, roots)

            # test critical points
            f_min = np.Inf
            x_sol = (x_min_bound + x_max_bound)/2 # defaults to bisection
            for crit_pt in crit_pts:
                if np.isreal(crit_pt) and crit_pt >= x_min_bound and crit_pt <= x_max_bound:
                    F_cp = np.polyval(coeff, crit_pt)
                    if np.isreal(F_cp) and F_cp < f_min:
                        x_sol = np.real(crit_pt)
                        f_min = np.real(F_cp)

    return x_sol


class Adam_ls(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, dtype=torch.float, debug=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, dtype=dtype, debug=debug)
        super(Adam_ls, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Adam_ls doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def __setstate__(self, state):
        super(Adam_ls, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def _numel(self):  # returns the total number of parameter elements
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self.param_groups[0]['params'], 0)  # reduce: continues operations on subsequent elements, see 'https://www.geeksforgeeks.org/reduce-in-python/'
        return self._numel_cache

    def _gather_flat_grad(self):  # return the (flattened) gradient (taken from LBFGS code)
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)  # flat gradient
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_data(self):
        views = []
        for p in self._params:
            if p.data is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)  # flat gradient
            views.append(view)
        return torch.cat(views, 0)

    def _copy_params(self):
        current_params = []
        for param in self._params:
            current_params.append(deepcopy(param.data))
        return current_params

    def _load_params(self, current_params):  # update the parameters
        i = 0
        for param in self._params:
            param.data[:] = current_params[i]
            i += 1

    def _add_update(self, step_size, update): # updates the model parameters by taking a step, theta_i <- theta_i + step_size*p_i (p: search  direction)
        offset = 0
        for p in self._params:
            numel = p.numel()  # number of elements
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))  # self.view_as(other) is equivalent to self.view(other.size())
            offset += numel
        assert offset == self._numel()  # If the expression is false, Python raises an AssertionError exception

    def step(self, options={}):
        """Performs a single optimization step.
        """

        group = self.param_groups[0]
        current_point = self._gather_flat_data()
        # print(current_point)
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            amsgrad = group['amsgrad']

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']

            state['step'] += 1

            if group['weight_decay'] != 0:
                grad.add_(group['weight_decay'], p.data)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(1 - beta1, grad) # mt, note inplace update => state['exp_avg'] is updated directly
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad) # mt, note inplace update => state['exp_avg_sq'] is updated directly
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sq.sqrt().add_(group['eps'])
            else:
                denom = exp_avg_sq.sqrt().add_(group['eps']) # sqrt(vt) + eps

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

            # print('hello')
            p.data.addcdiv_(-step_size, exp_avg, denom)


        # LINE SEARCH TIME!!
        dtype = group['dtype']
        debug = group['debug']

        # set search direction
        d = self._gather_flat_data()-current_point

        # closure evaluation counter
        closure_eval = 0

        # set initial step size
        t = 1

        # load options
        if(options):
            if('closure' not in options.keys()):
                raise(ValueError('closure option not specified.'))
            else:
                closure = options['closure']

            if('gtd' not in options.keys()):
                gtd = self._gather_flat_grad().dot(d)  # g^T*p (g times direction)
            else:
                gtd = options['gtd']

            if('eta' not in options.keys()):
                eta = 2  # steplength reduction factor
            elif(options['eta'] <= 0):
                raise(ValueError('Invalid eta; must be positive.'))
            else:
                eta = options['eta']

            if('c1' not in options.keys()):
                c1 = 1e-4
            elif(options['c1'] >= 1 or options['c1'] <= 0):
                raise(ValueError('Invalid c1; must be strictly between 0 and 1.'))
            else:
                c1 = options['c1']

            if('max_ls' not in options.keys()):
                max_ls = 10
            elif(options['max_ls'] <= 0):
                raise(ValueError('Invalid max_ls; must be positive.'))
            else:
                max_ls = options['max_ls']

            if('interpolate' not in options.keys()):
                interpolate = True
            else:
                interpolate = options['interpolate']

            if('inplace' not in options.keys()):
                inplace = False
            else:
                inplace = options['inplace']

            if('ls_debug' not in options.keys()):
                ls_debug = True
            else:
                ls_debug = options['ls_debug']

            if('increase_lr_on_min_ls' not in options.keys()):
                increase_lr_on_min_ls = 1.0
            else:
                increase_lr_on_min_ls = options['increase_lr_on_min_ls']

            if('decrease_lr_on_max_ls' not in options.keys()):
                decrease_lr_on_max_ls = 1.0
            else:
                decrease_lr_on_max_ls = options['decrease_lr_on_max_ls']

            if('line_search' not in options.keys()):
                line_search = True
            else:
                line_search = options['line_search']

        else:
            raise(ValueError('Options are not specified; need closure evaluating function.'))

        if line_search:
            self._add_update(t, -d) # reset parameters
            F_k = closure()
            closure_eval += 1


            # initialize values
            if(interpolate):
                if(torch.cuda.is_available()):
                    F_prev = torch.tensor(np.nan, dtype=dtype).cuda()
                else:
                    F_prev = torch.tensor(np.nan, dtype=dtype)

            ls_step = 0   # number of line search iterates
            t_prev = 0    # old steplength

            # begin print for debug mode
            if ls_debug:
                print('==================================== Begin Armijo line search ===================================')
                print('F(x): %.8e  g*d: %.8e' %(F_k, gtd))

            # check if search direction is descent direction
            if gtd >= 0:
                desc_dir = False
                if debug:
                    print('Not a descent direction!')
            else:
                desc_dir = True

            # store values if not in-place
            if not inplace:
                current_params = self._copy_params()

            # update and evaluate at new point
            self._add_update(t, d)  # update params: theta <- theta + t*d
            F_new = closure()  # evaluate the objective function
            closure_eval += 1

            # print info if debugging
            if(ls_debug):
                print('LS Step: %d  t: %.8e  F(x+td): %.8e  F-c1*t*g*d: %.8e  F(x): %.8e'
                      %(ls_step, t, F_new, F_k + c1*t*gtd, F_k))

            # check Armijo condition
            while F_new > F_k + c1*t*gtd or not is_legal(F_new):

                # check if maximum number of iterations reached
                if(ls_step >= max_ls):
                    if inplace:
                        self._add_update(-t, d)
                    else:
                        self._load_params(current_params)

                    group['lr'] *= decrease_lr_on_max_ls # decrease learning rate
                    t = 0
                    F_new = closure()
                    closure_eval += 1
                    fail = True
                    break

                else:
                    # store current steplength
                    t_new = t

                    # compute new steplength

                    # if first step or not interpolating, then multiply by factor
                    if(ls_step == 0 or not interpolate or not is_legal(F_new)):
                        t = t/eta

                    # if second step, use function value at new point along with
                    # gradient and function at current iterate
                    elif(ls_step == 1 or not is_legal(F_prev)):
                        t = polyinterp(np.array([[0, F_k.item(), gtd.item()], [t_new, F_new.item(), np.nan]]))

                    # otherwise, use function values at new point, previous point,
                    # and gradient and function at current iterate
                    else:
                        t = polyinterp(np.array([[0, F_k.item(), gtd.item()], [t_new, F_new.item(), np.nan],
                                                [t_prev, F_prev.item(), np.nan]]))

                    # if values are too extreme, adjust t
                    if(interpolate):
                        if(t < 1e-3*t_new):
                            t = 1e-3*t_new
                        elif(t > 0.6*t_new):
                            t = 0.6*t_new

                        # store old point
                        F_prev = F_new
                        t_prev = t_new

                    # update iterate and reevaluate
                    if inplace:
                        self._add_update(t-t_new, d)
                    else:
                        self._load_params(current_params)
                        self._add_update(t, d)  # update the model parameters, theta <- theta + t*d

                    F_new = closure()
                    closure_eval += 1
                    ls_step += 1  # iterate

                    # print info if debugging
                    if(ls_debug):
                        print('LS Step: %d  t: %.8e  F(x+td):   %.8e  F-c1*t*g*d: %.8e  F(x): %.8e'
                              %(ls_step, t, F_new, F_k + c1*t*gtd, F_k))

            if ls_step==0:
                group['lr'] *= increase_lr_on_min_ls # increase learning rate

            # print final steplength
            if ls_debug:
                print('Final Steplength:', t)
                print('===================================== End Armijo line search ====================================')

            return F_new, t * group['lr'], ls_step
        else:
            return closure(), group['lr'], 0
