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

########################################### Specify underlying  model ##################################################
# select true function
true_model = gpnets.gpnet1_1_4(sigma_f=1,lengthscale=0.1)

# set appr params
m = [100]  # nr of basis functions in each latent direction: Change this to add latent outputs
dim = len(m)
mt = np.prod(m)  # total nr of basis functions
tun = 4 # scaling parameter for L (nr of "std":s)
buildPhi_mod = gprh.buildPhi(m,type='point',tun=tun)
nprior = 500
xprior = torch.linspace(0,1, nprior).unsqueeze(-1)
phi, sq_lambda, _ = buildPhi_mod.getphi( true_model, m, nprior, mt, train_x=xprior )

sigma_f = torch.exp(true_model.gp.log_sigma_f)
lengthscale = torch.exp(true_model.gp.log_lengthscale)

lprod=torch.ones(1)
omega_sum=torch.zeros(mt,1)
for q in range(dim):
    lprod*=lengthscale[q].pow(2)
    omega_sum+=lengthscale[q].pow(2)*sq_lambda[:,q].view(mt,1).pow(2)

lambda_diag = torch.pow(lprod, 0.5).mul(torch.exp( omega_sum.mul(0.5).neg() )).mul(math.pow(2.0*math.pi,dim/2.0)).view(mt).mul(sigma_f.pow(2))

Ka = phi.mm(lambda_diag.diag()).mm(phi.t()) + torch.eye(nprior).mul( 1e-4 )

L = torch.cholesky(Ka,upper=False)
sample = L.mm(torch.randn(nprior,1)).detach().flatten()

# # plot the function
# fplot, ax = plt.subplots(1, 1, figsize=(4, 3))
# # Plot true function as solid black
# ax.plot(xprior.detach().numpy(), sample.numpy(), 'k')
########################################################################################################################

# use integral or point measurements
# integral = False
points = True

if points:
    meastype = 'point'

    n = 200
    indices = (nprior*np.random.rand(n)).astype(int)
    train_x = torch.from_numpy( xprior.numpy()[indices] )
    train_y = torch.from_numpy( sample.numpy()[indices].flatten() )
    noise_std = train_y.abs().var().mul( 0.1 )
    train_y = train_y.add( torch.randn(n).mul(noise_std) )

    nt=nprior
    test_x = torch.linspace(0, 1, nt).view(nt,1)

    # dom_points
    dom_points = test_x

m = [50]  # nr of basis functions in each latent direction: Change this to add latent outputs
dim = len(m)
mt = np.prod(m)  # total nr of basis functions
tun = 4 # scaling parameter for L (nr of "std":s)

# select model
model = gpnets.gpnet1_1_3(sigma_f=1,lengthscale=1,sigma_n=1)  # assumed model
print('Number of parameters: %d' %model.npar)

# loss function
lossfu = gprh.NegLOOCrossValidation_phi_noBackward()
# lossfu = gprh.NegMarginalLogLikelihood_phi_noBackward()

# buildPhi object
tun = 4 # scaling parameter for L (nr of "std":s)
buildPhi = gprh.buildPhi(m,type=meastype,tun=tun)

# optimiser
optimiser = FullBatchLBFGS(model.parameters(), lr=1, history_size=10)

# closure: should return the loss
def closure():
    global L
    optimiser.zero_grad() # zero gradients
    phi, sq_lambda, L = buildPhi.getphi(model,m,n,mt,train_x=train_x,dom_points=dom_points)
    return lossfu(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi, train_y, sq_lambda)


loss = closure()  # compute initial loss

training_iterations = 50
for i in range(training_iterations):
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
with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'r*')

    # Plot true function as solid black
    ax.plot(test_x.numpy(), sample.numpy(), 'k')

    # plot 95% credibility region
    upper = torch.squeeze(test_f, 1) + 2*cov_f.pow(0.5)
    lower = torch.squeeze(test_f, 1) - 2*cov_f.pow(0.5)
    ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

    # plot predictions
    ax.plot(test_x.numpy(), test_f.detach().numpy(), 'b')

    # plot latent outputs
    train_m = model(test_x)
    for w in range(dim):
        ax.plot(test_x.numpy(), train_m[:,w].numpy(), 'g')

    #ax.set_ylim([-2, 2])
    #ax.legend(['Observed Data', 'True', 'Predicted'])
    plt.show()
