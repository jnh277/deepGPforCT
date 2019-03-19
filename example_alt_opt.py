import torch
import math
from matplotlib import pyplot as plt
import gp_regression_hilbert as gprh
import gpnets

# add path to optimisation code
import sys
sys.path.append('./opti_functions/')
from Adam_ls import Adam_ls
from LBFGS import FullBatchLBFGS

def truefunc(omega,points):
    # return torch.cos(torch.squeeze(points, 1) * omega)
    out = points.clone()
    out[points<=0.5] = -1
    out[points>0.5] = 1
    return out.view(-1)

omega = 2.0*math.pi
noise_std = 0.05
n = 50
train_x = torch.linspace(0,1, n).view(n,1)
train_y = truefunc(omega,train_x) + torch.randn(n) * noise_std

nt = 1000
test_x = torch.linspace(0, 1, nt).view(nt,1)

# set appr params
m = 60 # nr of basis functions

# select net
model = gpnets.gpnet1_1_2(sigma_f=1, lengthscale=1, sigma_n=1)

nLL = gprh.NegMarginalLogLikelihood_noBackward()  # this is the loss function

##### pick optimiser
# optimiser = FullBatchLBFGS(model.parameters(), lr=1, history_size=2, debug=True)  # full-batch L-BFGS optimizer
optimiser = Adam_ls(model.parameters(), lr=1)  # Adam with line search

# closure: should return the loss
def closure():
    train_m = model(train_x)
    loss = nLL(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, train_m, train_y, m)
    return loss

loss = closure() # compute initial loss

training_iterations = 100
for i in range(training_iterations):

    options = {'line_search': True, 'closure': closure, 'max_ls': 5, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.5, 'increase_lr_on_min_ls': 2}

    optimiser.zero_grad() # zero gradients
    loss.backward() # propagate derivatives
    loss, lr, ls_iters = optimiser.step(options=options) # compute new loss

    print('Iter %d/%d - Loss: %.3f - LR: %.8f - LS iterates: %0.0f  - sigma_f: %0.3f - lengthscale: %.3f - sigma_n: %.3f' % (
    i + 1, training_iterations, loss.item(), lr, ls_iters,
    model.gp.log_sigma_f.exp().item(),
    model.gp.log_lengthscale.exp().item(),
    model.gp.log_sigma_n.exp().item()
    ))

# now make predictions
test_f, cov_f = model(train_x, train_y, m, test_x)

with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'r*')

    # Plot true function as solid black
    ax.plot(test_x.numpy(), truefunc(omega,test_x).numpy(), 'k')

    # plot h
    train_m = model(test_x)
    ax.plot(test_x.numpy(), train_m.numpy(), 'g')

    # plot 95% credibility region
    upper = torch.squeeze(test_f, 1) + 2*cov_f.pow(0.5)
    lower = torch.squeeze(test_f, 1) - 2*cov_f.pow(0.5)
    ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

    # plot predictions
    ax.plot(test_x.numpy(), test_f.numpy(), 'b')

    plt.show()
