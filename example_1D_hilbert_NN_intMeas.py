import torch
import math
import gp_regression_hilbert as gprh
import gpnets
import integration as intgr
import numpy as np
import time

# select true function
truefunc=gprh.step
truefunc_int=gprh.step_int

# use integral or point measurements
integral=True
points=not(integral)

if integral:
    meastype='int'

    omega=4*math.pi
    noise_std=0.001
    n = 50

    # inputs (integral limits)
    train_x=torch.rand(n,2)

    # output (integral measurements)
    train_y = truefunc_int(omega,train_x) + torch.randn(n) * noise_std

    # test points
    nt=1000
    test_x = torch.linspace(0, 1, nt).view(nt,1)

if points:
    meastype='point'

    omega=4*math.pi
    noise_std=0.001
    n = 150
    train_x = torch.Tensor(n, 1)
    train_x[:, 0] = torch.linspace(0,1, n)
    train_y = truefunc(omega,train_x) + torch.randn(n) * noise_std

    nt=1000
    test_x = torch.linspace(0, 1, nt).view(nt,1)

# set appr params
m = [50] # nr of basis functions in each latent direction: Change this to add latent outputs
mt= np.prod(m) # total nr of basis functions

# select model
model = gpnets.gpnet1_2_1()

# loss function
nLL = gprh.NegMarginalLogLikelihood_phi_noBackward()

# buildPhi object
int_method = 3 # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
ni = 400 # nr of intervals
tun=4
buildPhi = gprh.buildPhi(m,type=meastype,ni=ni,int_method=int_method,tun=tun)

# optimiser
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
],lr=0.005)

# scheduler for optimisation
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.99,min_lr=1e-6)

training_iterations = 500
for i in range(training_iterations):
    # Zero backprop gradients
    optimizer.zero_grad()

    # build phi
    phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,train_x)

    # Calc loss
    loss = nLL(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi.view(n*mt,1), train_y, sq_lambda)

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
test_f, cov_f = model(None, train_y, phi, sq_lambda, L, test_x)

# plot
gprh.makeplot(model,train_x,train_y,test_x,test_f,cov_f,truefunc,omega,len(m),meastype=meastype)
