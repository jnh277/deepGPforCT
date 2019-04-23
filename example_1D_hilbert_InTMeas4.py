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
    n = 150
    train_x = torch.linspace(0,1, n).unsqueeze(-1)
    train_y = truefunc(train_x) + torch.randn(n) * noise_std

    nt = 1000
    test_x = torch.linspace(0, 1, nt).view(nt,1)

    # dom_points
    dom_points = test_x


# STEP 1
# set appr params
m2 = [150]
tun2 = 30
diml = len(m2)  # nr of latent outputs

training_iterations = 200

regweight = 0.0001

mybestnet = gpnets.gpnet1_1_4(sigma_f=1,lengthscale=[1],sigma_n=1,covfunc=gprh.covfunc(type='matern',nu=2.5))
optimiser2 = FullBatchLBFGS(mybestnet.parameters(), lr=1, history_size=10)

# STEP 3
int_method = 2  # 1) trapezoidal, 2) simpsons standard, 3) simpsons 3/8
ni = 200  # nr of intervals in numerical integration

training_iterations2 = 80

regweight2 = 0.1

# optimiser
optimiser3 = FullBatchLBFGS(mybestnet.parameters(), lr=1, history_size=10)
lossfu3 = gprh.NegLOOCrossValidation_phi_noBackward(mybestnet.gp.covfunc)
# lossfu3 = gprh.NegMarginalLogLikelihood_phi_noBackward(mybestnet.gp.covfunc)





#######
#       Pre-train the net directly on data
#######
buildPhi = gprh.buildPhi([0],type=meastype,ni=ni,int_method=int_method)

if points:
    def closure2():
        # optimiser2.zero_grad() # zero gradients
        return (mybestnet(train_x)[:,0].unsqueeze(-1) .sub( train_y.unsqueeze(-1) ) ).pow(2) .sum() .div(n) + gprh.regulariser(mybestnet,weight=regweight)
else:
    def closure2():
        # optimiser2.zero_grad() # zero gradients
        ints = torch.zeros(n)
        for q in range(n):
            a = train_x[q,0]
            b = train_x[q,1]
            h = (b-a)/ni

            zz = mybestnet( torch.linspace(a,b,ni+1).view(ni+1,1) )[:,0].unsqueeze(-1)

            ints[q] = torch.sum(zz*buildPhi.sc.t() ).mul(buildPhi.fact*h)
        return  (ints.sub( train_y ) ).pow(2) .sum() .div(n) + gprh.regulariser(mybestnet,weight=regweight)

loss2 = closure2()

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
    ax.plot(test_x.numpy(), mybestnet(test_x).detach().numpy(), 'r')

    plt.show()


#######
#       Train the joint model
######
buildPhi2 = gprh.buildPhi(m2,type=meastype,ni=ni,int_method=int_method,tun=tun2)

def closure3():
    global L
    phi, sq_lambda, L = buildPhi2.getphi(mybestnet,n,train_x=train_x,dom_points=dom_points)
    return lossfu3(mybestnet.gp.log_sigma_f, mybestnet.gp.log_lengthscale, mybestnet.gp.log_sigma_n, phi, train_y, sq_lambda) + gprh.regulariser(mybestnet,weight=regweight2)

loss3 = closure3()

for i in range(training_iterations2):
    options = {'closure': closure3, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser3.zero_grad() # zero gradients
    loss3.backward() # propagate derivatives
    loss3, lr, ls_iters = optimiser3.step(options=options) # compute new loss

    gprh.optiprint(i,training_iterations2,loss3.item(),lr,ls_iters,mybestnet,L)


# update phi
phi,sq_lambda,L = buildPhi2.getphi(mybestnet,n,train_x=train_x,dom_points=dom_points)

# now make predictions
test_f, cov_f = mybestnet(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)

# torch.save((model,train_x,train_y,test_x,test_f,cov_f,truefunc,omega,diml,meastype),'mymodel')
# model,train_x,train_y,test_x,test_f,cov_f,truefunc,omega,diml,meastype = torch.load('mymodel')

# plot
gprh.makeplot(mybestnet,train_x,train_y,test_x,test_f,cov_f,truefunc,diml,meastype=meastype)
