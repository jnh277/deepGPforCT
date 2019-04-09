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
truefunc = gprh.stepsin
truefunc_int = gprh.step_int

# use integral or point measurements
integral = False
points = not integral

if integral:
    meastype = 'int'

    noise_std = 0.001
    n = 50

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

    nt = 1400
    test_x = torch.linspace(0, 1, nt).view(nt,1)

    # dom_points
    dom_points = test_x

# set appr params
m = [50]  # nr of basis functions in each latent direction: Change this to add latent outputs
diml = len(m)  # nr of latent outputs
mt = np.prod(m)  # total nr of basis functions

# select model
model = gpnets.gpnet1_1_1(sigma_f=1,lengthscale=1,sigma_n=1)
print('Number of parameters: %d' %model.npar)

# loss function
lossfu = gprh.NegLOOCrossValidation_phi_noBackward()
# lossfu = gprh.NegMarginalLogLikelihood_phi_noBackward()

# buildPhi object
int_method = 1  # 1) trapezoidal, 2) simpsons standard, 3) simpsons 3/8
ni = 200  # nr of intervals in numerical integration
tun = 4 # scaling parameter for L (nr of "std":s)
buildPhi = gprh.buildPhi(m,type=meastype,ni=ni,int_method=int_method,tun=tun)

# optimiser
optimiser = FullBatchLBFGS(model.parameters(), lr=1, history_size=10)

# closure: should return the loss
def closure():
    global L
    optimiser.zero_grad() # zero gradients
    phi, sq_lambda, L = buildPhi.getphi(model,m,n,mt,train_x=train_x,dom_points=dom_points)
    return lossfu(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi, train_y, sq_lambda)


loss = closure()  # compute initial loss

training_iterations = 250
for i in range(training_iterations):
    # model.scale+=1.01
    # model.scale2*=1.2
    options = {'closure': closure, 'max_ls': 5, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser.zero_grad() # zero gradients
    loss.backward() # propagate derivatives
    loss, lr, ls_iters = optimiser.step(options=options) # compute new loss

    # print
    gprh.optiprint(i,training_iterations,loss.item(),lr,ls_iters,model,L)

# update phi
phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,train_x=train_x,dom_points=dom_points)

# now make predictions
test_f, cov_f = model(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)

# torch.save((model,train_x,train_y,test_x,test_f,cov_f,truefunc,omega,diml,meastype),'mymodel')
# model,train_x,train_y,test_x,test_f,cov_f,truefunc,omega,diml,meastype = torch.load('mymodel')

# plot
gprh.makeplot(model,train_x,train_y,test_x,test_f,cov_f,truefunc,diml,meastype=meastype)




test_f = test_f.detach()

# now train a neural net, hahaha!!!!
# optimiser
mybestnet = gpnets.gpnet1_1_2(sigma_f=1,lengthscale=1,sigma_n=1)

optimiser2 = FullBatchLBFGS(mybestnet.parameters(), lr=1, history_size=10)

def closure2():
    optimiser2.zero_grad() # zero gradients
    return (mybestnet(test_x) .sub( test_f ) ).pow(2) .sum() # todo: CV approach here to avoid overfit

loss2 = closure2()

training_iterations =350
for i in range(training_iterations):
    options = {'closure': closure2, 'max_ls': 5, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser2.zero_grad() # zero gradients
    loss2.backward() # propagate derivatives
    loss2, lr, ls_iters = optimiser2.step(options=options) # compute new loss

    # print
    print(loss2.item())


with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

    # plot predictions
    ax.plot(test_x.numpy(), test_f.numpy(), 'b')
    ax.plot(test_x.numpy(), mybestnet(test_x).detach().numpy(), 'r')

    plt.show()



# now train a gp again, hohoho!!!!
# optimiser
# mybestnet = gpnets.gpnet1_1_2(sigma_f=1,lengthscale=1,sigma_n=1)

buildPhi2 = gprh.buildPhi(m,type=meastype,ni=ni,int_method=int_method,tun=tun)

optimiser3 = FullBatchLBFGS(mybestnet.parameters(), lr=1, history_size=10)
lossfu3 = gprh.NegLOOCrossValidation_phi_noBackward()

def closure3():
    global L
    optimiser3.zero_grad() # zero gradients
    phi, sq_lambda, L = buildPhi2.getphi(mybestnet,m,n,mt,train_x=train_x,dom_points=dom_points)
    return lossfu3(mybestnet.gp.log_sigma_f, mybestnet.gp.log_lengthscale, mybestnet.gp.log_sigma_n, phi, train_y, sq_lambda)

loss3 = closure3()

training_iterations = 450
for i in range(training_iterations):
    options = {'closure': closure3, 'max_ls': 5, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser3.zero_grad() # zero gradients
    loss3.backward() # propagate derivatives
    loss3, lr, ls_iters = optimiser3.step(options=options) # compute new loss

    gprh.optiprint(i,training_iterations,loss3.item(),lr,ls_iters,mybestnet,L)


# update phi
phi,sq_lambda,L = buildPhi2.getphi(mybestnet,m,n,mt,train_x=train_x,dom_points=dom_points)

# now make predictions
test_f, cov_f = mybestnet(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)

# torch.save((model,train_x,train_y,test_x,test_f,cov_f,truefunc,omega,diml,meastype),'mymodel')
# model,train_x,train_y,test_x,test_f,cov_f,truefunc,omega,diml,meastype = torch.load('mymodel')

# plot
gprh.makeplot(mybestnet,train_x,train_y,test_x,test_f,cov_f,truefunc,diml,meastype=meastype)
