import torch
import math
import gp_regression_hilbert as gprh
import gpnets
import integration as intgr
import numpy as np
import time
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('./opti_functions/')
from Adam_ls import Adam_ls
from LBFGS import FullBatchLBFGS

# select true function
truefunc = gprh.step
truefunc_int = gprh.step_int

# use integral or point measurements
integral = True
points = not integral

# torch.manual_seed(0)

if integral:
    meastype = 'int'

    noise_std = 0.001
    n = 100

    # inputs (integral limits)
    train_x = torch.rand(n,2)

    # output (integral measurements)
    train_y = truefunc_int(train_x) + torch.randn(n) * noise_std

    # test points
    nt = 1000
    test_x = torch.linspace(0, 1, nt).view(nt,1)

    # dom_points
    dom_points = test_x

if points:
    meastype = 'point'

    noise_std = 0.01
    n = 100
    train_x = torch.linspace(0,1, n).unsqueeze(-1)
    train_y = truefunc(train_x) + torch.randn(n) * noise_std

    nt = 1000
    test_x = torch.linspace(0, 1, nt).view(nt,1)

    # dom_points
    dom_points = test_x


## STEP 1
# set appr params
m = [90]  # nr of basis functions in each latent direction: Change this to add latent outputs
diml = len(m)  # nr of latent outputs

# select model
model = gpnets.gpnet1_1_1(sigma_f=1,lengthscale=[1],sigma_n=1,covfunc=gprh.covfunc('matern',nu=2.5))
tun = 30 # scaling parameter for L

print('Number of parameters: %d' %model.npar)

# loss function
lossfu = gprh.NegLOOCrossValidation_phi_noBackward(model.gp.covfunc)
# lossfu = gprh.NegMarginalLogLikelihood_phi_noBackward(model.gp.covfunc)

# optimiser
optimiser = FullBatchLBFGS(model.parameters(), lr=1, history_size=10)

# STEP 2/3
mybestnet = gpnets.gpnet1_2_2(sigma_f=1,lengthscale=[1,1],sigma_n=1)
optimiser2 = FullBatchLBFGS(mybestnet.parameters(), lr=1, history_size=10)

# STEP 3
int_method = 3  # 1) trapezoidal, 2) simpsons standard, 3) simpsons 3/8
ni = 200  # nr of intervals in numerical integration
m2 = [10,10]
tun2 = 5

optimiser3 = FullBatchLBFGS(mybestnet.parameters(), lr=1, history_size=10)
lossfu3 = gprh.NegLOOCrossValidation_phi_noBackward(mybestnet.gp.covfunc)
# lossfu3 = gprh.NegMarginalLogLikelihood_phi_noBackward(mybestnet.gp.covfunc)


################################## step 1
buildPhi = gprh.buildPhi(m,type=meastype,tun=tun,ni=ni,int_method=int_method)

# closure: should return the loss
def closure():
    global L
    optimiser.zero_grad() # zero gradients
    phi, sq_lambda, L = buildPhi.getphi(model,n,train_x=train_x,dom_points=dom_points)
    return lossfu(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi, train_y, sq_lambda)
loss = closure()  # compute initial loss

training_iterations = 15
for i in range(training_iterations):
    # model.scale+=1.01
    # model.scale2*=1.2
    options = {'closure': closure, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser.zero_grad() # zero gradients
    loss.backward() # propagate derivatives
    loss, lr, ls_iters = optimiser.step(options=options) # compute new loss

    # print
    gprh.optiprint(i,training_iterations,loss.item(),lr,ls_iters,model,L)

# update phi
phi,sq_lambda,L = buildPhi.getphi(model,n,train_x=train_x,dom_points=dom_points)

# now make predictions
test_f, cov_f = model(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)

# plot
gprh.makeplot(model,train_x,train_y,test_x,test_f,cov_f,truefunc,diml,meastype=meastype)

test_f = test_f.detach()


################################## step 2
# now train a neural net

def closure2():
    optimiser2.zero_grad() # zero gradients
    return (mybestnet(test_x)[:,0].unsqueeze(-1) .sub( test_f ) ).pow(2) .sum() .div(test_f.numel())

loss2 = closure2()

training_iterations =200
for i in range(training_iterations):
    options = {'closure': closure2, 'max_ls': 5, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser2.zero_grad() # zero gradients
    loss2.backward() # propagate derivatives
    loss2, lr, ls_iters = optimiser2.step(options=options) # compute new loss

    # print
    print(i,loss2.item())


with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

    # plot predictions
    ax.plot(test_x.numpy(), test_f.numpy(), 'b')
    ax.plot(test_x.numpy(), mybestnet(test_x).detach().numpy(), 'r')

    plt.show()


################################## step 3
# now train a gp again
buildPhi2 = gprh.buildPhi(m2,type=meastype,ni=ni,int_method=int_method,tun=tun2)

def closure3():
    global L
    phi, sq_lambda, L = buildPhi2.getphi(mybestnet,n,train_x=train_x,dom_points=dom_points)
    return lossfu3(mybestnet.gp.log_sigma_f, mybestnet.gp.log_lengthscale, mybestnet.gp.log_sigma_n, phi, train_y, sq_lambda)

loss3 = closure3()

training_iterations = 80
for i in range(training_iterations):
    options = {'closure': closure3, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser3.zero_grad() # zero gradients
    loss3.backward() # propagate derivatives
    loss3, lr, ls_iters = optimiser3.step(options=options) # compute new loss

    gprh.optiprint(i,training_iterations,loss3.item(),lr,ls_iters,mybestnet,L)


# update phi
phi,sq_lambda,L = buildPhi2.getphi(mybestnet,n,train_x=train_x,dom_points=dom_points)

# now make predictions
test_f, cov_f = mybestnet(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)

# plot
gprh.makeplot(mybestnet,train_x,train_y,test_x,test_f,cov_f,truefunc,diml,meastype=meastype)
