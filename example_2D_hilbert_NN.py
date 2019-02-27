import torch
import math
import gp_regression_hilbert as gprh
import gpnets
import integration as intgr
import numpy as np
import time

# select true function
truefunc=gprh.circlefunc
# truefunc_int=gprh.step_int

# use integral or point measurements
integral=False
points=not(integral)

if integral:
    meastype='int'

if points:
    meastype='point'

    noise_std=0.05
    n = 3000

    train_x = -1+2*torch.rand(n,2)#torch.cat((train_r*torch.cos(train_theta),train_r*torch.sin(train_theta)),1)

    train_y = truefunc(train_x) + torch.randn(n) * noise_std

    ntx=200
    nty=200
    nt=ntx*nty
    X = np.linspace(-1, 1, ntx)
    Y = np.linspace(-1, 1, nty)
    X,Y = np.meshgrid(X, Y)
    test_x = torch.from_numpy(np.concatenate((np.reshape(X,(nt,1)),np.reshape(Y,(nt,1))),axis=1)).float()

# set appr params
m = [40] # nr of basis functions in each latent direction: Change this to add latent outputs
diml=len(m) # nr of latent outputs
mt= np.prod(m) # total nr of basis functions

# select model
model = gpnets.gpnet2_1_1(sigma_n=noise_std)

# loss function
nLL = gprh.NegMarginalLogLikelihood_phi_noBackward()

# buildPhi object
int_method=3 # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
ni=400 # nr of intervals in numerical integration
tun=4 # scaling parameter for L (nr of "std":s)
buildPhi = gprh.buildPhi(m,type=meastype,ni=ni,int_method=int_method,tun=tun)

# optimiser
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
],lr=0.01)

# scheduler for optimisation
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.99,min_lr=1e-6)

training_iterations = 3000
for i in range(training_iterations):
    # Zero backprop gradients
    optimizer.zero_grad()

    # build phi
    phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,train_x)

    # Calc loss
    loss = nLL(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi, train_y, sq_lambda)

    # Backprop derivatives
    loss.backward()

    # step forward
    optimizer.step()
    scheduler.step(i)

    # print
    gprh.optiprint(i,training_iterations,loss.item(),model,L)

# update phi
phi,_,_ = buildPhi.getphi(model,m,n,mt,train_x)

# now make predictions
test_f, cov_f = model(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)

# torch.save((model,train_x,train_y,test_x,test_f,cov_f,truefunc,omega,diml,meastype),'mymodel')
# model,train_x,train_y,test_x,test_f,cov_f,truefunc,omega,diml,meastype = torch.load('mymodel')

# plot
gprh.makeplot2D(model,X,Y,ntx,nty,train_x,test_x,test_f,cov_f,truefunc,diml,vmin=0,vmax=1)
