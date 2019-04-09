import torch
import gp_regression_hilbert as gprh
import gpnets
import numpy as np
import radon_data as rd
import math
from time import time

from matplotlib import pyplot as plt
from matplotlib import cm

# add path to optimisation code
import sys
sys.path.append('./opti_functions/')
from Adam_ls import Adam_ls
from LBFGS import FullBatchLBFGS

# use integral or point measurements
integral = False
point = not(integral)

if integral:
    meastype='int'

    # noise level
    noise_std = 0.001

    nmeas_proj = 5
    nproj = 20
    n = nmeas_proj*nproj

    # import data
    dataname = 'circle_square'
    print('Getting data...')
    train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp = rd.getdata(dataname=dataname,image_res=3e3,nmeas_proj=nmeas_proj,
                                                                           nproj=nproj,nt=5*np.int(np.sqrt(6*n)),sigma_n=noise_std,reconstruct_fbp=True)
    print('Data ready!')

    # convert
    train_y = torch.from_numpy(train_y).float()
    x0 = torch.from_numpy(x0).float()
    unitvecs = torch.from_numpy(unitvecs).float()

    # # normalise
    # train_y = train_y.sub(train_y.mean())
    # train_y = train_y.div(train_y.abs().max())

    # test points
    ntx=np.size(X,1)
    nty=np.size(X,0)
    test_x = torch.from_numpy(np.concatenate((np.reshape(X,(np.size(X),1)),np.reshape(Y,(np.size(X),1))),axis=1)).float()

    # domain random points (used to limit L from below)
    ndx=100
    ndy=100
    nd=ndx*ndy
    Xd = np.linspace(-Rlim, Rlim, ndx)
    Yd = np.linspace(-Rlim, Rlim, ndy)
    Xd,Yd = np.meshgrid(Xd, Yd)
    dom_points = torch.from_numpy(np.concatenate((np.reshape(Xd,(nd,1)),np.reshape(Yd,(nd,1))),axis=1)).float()

if point:
    meastype='point'

    # # select true function
    # truefunc=gprh.circlefunc2

    noise_std = 0.001
    n = 1000

    # import data
    dataname = 'phantom'
    print('Getting data...')
    train_y, train_x, X, Y, Z = rd.getdata(dataname=dataname,image_res=3e3,nt=np.max((np.int(np.sqrt(10*n)),np.int(np.sqrt(5000)))),sigma_n=noise_std,points=True,npmeas=n)
    print('Data ready!')

    # convert
    train_y = torch.from_numpy(train_y).float()
    train_x = torch.from_numpy(train_x).float()

    # # normalise
    # train_y = train_y.sub(train_y.mean())
    # train_y = train_y.div(train_y.abs().max())

    # test points
    ntx = np.size(X,1)
    nty = np.size(X,0)
    test_x = torch.from_numpy(np.concatenate((np.reshape(X,(np.size(X),1)),np.reshape(Y,(np.size(X),1))),axis=1)).float()

    # # input data
    # train_x = torch.rand(n,2)
    # train_y = truefunc(train_x) + torch.randn(n) * noise_std
    #
    # # test point
    # ntx=200
    # nty=200
    # nt=ntx*nty
    # X = np.linspace(0, 1, ntx)
    # Y = np.linspace(0, 1, nty)
    # X,Y = np.meshgrid(X, Y)
    # test_x = torch.from_numpy(np.concatenate((np.reshape(X,(nt,1)),np.reshape(Y,(nt,1))),axis=1)).float()

    # domain random points (used to limit L from below)
    ndx = 100
    ndy = 100
    nd = ndx*ndy
    Xd = np.linspace(-1, 1, ndx)
    Yd = np.linspace(-1, 1, ndy)
    Xd,Yd = np.meshgrid(Xd, Yd)
    dom_points = torch.from_numpy(np.concatenate((np.reshape(Xd,(nd,1)),np.reshape(Yd,(nd,1))),axis=1)).float()

# set appr params
m = [40,40]  # nr of basis functions in each latent direction: Change this to add latent outputs
diml = len(m)  # nr of latent outputs
mt = np.prod(m)  # total nr of basis functions

# select model
# torch.manual_seed(9202720329292250078) # circle_square, gpnet2_1_8, noise_std = 0.001, n = 1000
# torch.manual_seed(2)
model_basic = gpnets.gpnet2_2_1(sigma_f=1, lengthscale=[1,1], sigma_n=1)
print('Number of parameters: %d' %model_basic.npar)

# loss function
lossfu_basic = gprh.NegLOOCrossValidation_phi_noBackward()

# buildPhi object
tun=4 # scaling parameter for L (nr of "std":s)
if integral:
    int_method = 2  # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
    ni = 200  # nr of intervals in numerical integration
    buildPhi_basic = gprh.buildPhi(m,type=meastype,ni=ni,int_method=int_method,tun=tun,x0=x0,unitvecs=unitvecs,Rlim=Rlim)
else:
    buildPhi_basic = gprh.buildPhi(m,type=meastype,tun=tun)


# optimiser
optimiser_basic = FullBatchLBFGS(model_basic.parameters(), lr=1, history_size=30)
# optimiser = Adam_ls(model.parameters(), lr=0.001)

# regularizer todo: build this into the optimiser
def regularizer(model):
    reg = torch.zeros(1)
    for p in model.parameters():
        reg = reg.add(  p.pow(2).sum()  )
    return reg

# compute initial loss
def closure_basic():
    optimiser_basic.zero_grad()
    global L
    if integral:
        phi, sq_lambda, L = buildPhi_basic.getphi(model_basic,m,n,mt,dom_points)
    else:
        phi, sq_lambda, L = buildPhi_basic.getphi(model_basic,m,n,mt,dom_points,train_x=train_x)
    return lossfu_basic(model_basic.gp.log_sigma_f, model_basic.gp.log_lengthscale, model_basic.gp.log_sigma_n, phi, train_y, sq_lambda) #+ regularizer(model).mul(1)
loss_basic = closure_basic()

saveFreq = 200 # how often do you wanna save the model?

training_iterations = 250

for i in range(training_iterations):

    options = {'line_search': True, 'closure': closure_basic, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser_basic.zero_grad() # zero gradients

    loss_basic.backward()  # Backprop derivatives

    loss_basic, lr, ls_step = optimiser_basic.step(options=options) # compute new loss

    # print
    gprh.optiprint(i, training_iterations, loss_basic.item(), lr, ls_step, model_basic, L)

# update phi
if integral:
    phi,sq_lambda,L = buildPhi_basic.getphi(model_basic,m,n,mt,dom_points)
else:
    phi,sq_lambda,L = buildPhi_basic.getphi(model_basic,m,n,mt,dom_points,train_x=train_x)

# now make predictions
test_f, cov_f = model_basic(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)


#########################################################
# STEP 2
#########################################################
test_f = test_f.detach()

# now train a neural net, hahaha!!!!
# optimiser
model = gpnets.gpnet2_2_3(sigma_f=1,lengthscale=[1,1],sigma_n=1)

optimiser2 = FullBatchLBFGS(model.parameters(), lr=1, history_size=10)

def closure2():
    optimiser2.zero_grad() # zero gradients
    return (model(test_x) .sub( test_f ) ).pow(2) .sum()

loss2 = closure2()

training_iterations =3000
for i in range(training_iterations):
    options = {'closure': closure2, 'max_ls': 5, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser2.zero_grad() # zero gradients
    loss2.backward() # propagate derivatives
    loss2, lr, ls_iters = optimiser2.step(options=options) # compute new loss

    # print
    print(i,loss2.item())


#########################################################
# STEP 3
#########################################################
# buildPhi object
tun=4 # scaling parameter for L (nr of "std":s)
if integral:
    int_method = 2  # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
    ni = 200  # nr of intervals in numerical integration
    buildPhi = gprh.buildPhi(m,type=meastype,ni=ni,int_method=int_method,tun=tun,x0=x0,unitvecs=unitvecs,Rlim=Rlim)
else:
    buildPhi = gprh.buildPhi(m,type=meastype,tun=tun)

# loss function
lossfu = gprh.NegLOOCrossValidation_phi_noBackward()

# optimiser
optimiser = FullBatchLBFGS(model.parameters(), lr=1, history_size=30)
# optimiser = Adam_ls(model.parameters(), lr=0.001)

# regularizer todo: build this into the optimiser
def regularizer(model):
    reg = torch.zeros(1)
    for p in model.parameters():
        reg = reg.add(  p.pow(2).sum()  )
    return reg

# compute initial loss
def closure():
    optimiser.zero_grad()
    global L
    if integral:
        phi, sq_lambda, L = buildPhi.getphi(model,m,n,mt,dom_points)
    else:
        phi, sq_lambda, L = buildPhi.getphi(model,m,n,mt,dom_points,train_x=train_x)
    return lossfu(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi, train_y, sq_lambda) #+ regularizer(model).mul(1)
loss = closure()

saveFreq = 200 # how often do you wanna save the model?

training_iterations = 850

for i in range(training_iterations):

    options = {'line_search': True, 'closure': closure, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser.zero_grad() # zero gradients

    loss.backward()  # Backprop derivatives

    loss, lr, ls_step = optimiser.step(options=options) # compute new loss

    # print
    gprh.optiprint(i, training_iterations, loss.item(), lr, ls_step, model, L)

# update phi
if integral:
    phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,dom_points)
else:
    phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,dom_points,train_x=train_x)

# now make predictions
test_f, cov_f = model(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)



#########################################################
# PLOT
#########################################################
vmin = -1
vmax = 1
if meastype=='point':
    fplot, ax = plt.subplots(2, 3, figsize=(27,9))

    ## true function & meas
    # Z = np.reshape(truefunc(test_x).numpy(),(ntx,nty))
    pc = ax[0,0].pcolor(X,Y,Z, cmap=cm.coolwarm)
    pc.set_clim(vmin,vmax)
    ax[0,0].plot(train_x[:,0].numpy(),train_x[:,1].numpy(),'ro', alpha=0.3)

    ## prediction
    Zp = np.reshape(test_f.detach().numpy(),(ntx,nty))
    pc = ax[0,1].pcolor(X,Y,Zp, cmap=cm.coolwarm)
    pc.set_clim(vmin,vmax)

    ## covariance
    Zc = np.reshape(cov_f.detach().numpy(),(ntx,nty))
    pc = ax[0,2].pcolor(X,Y,Zc, cmap=cm.coolwarm)
    pc.set_clim(vmin,vmax)

    # plot latent outputs
    train_m = model(test_x)
    for w in range(diml):
        Zm = np.reshape(train_m[:,w].detach().numpy(),(ntx,nty))
        pc = ax[1,w].pcolor(X,Y,Zm, cmap=cm.coolwarm)
        pc.set_clim(vmin,vmax)

    ## shared colorbar
    fplot.colorbar(pc, ax=ax.ravel().tolist())

    plt.show()
else:
    fplot, ax = plt.subplots(2, 3, figsize=(27,9))

    ## true function & meas
    pc = ax[0,0].pcolor(X,Y,Z, cmap=cm.coolwarm)
    pc.set_clim(vmin,vmax)

    ## prediction
    Zp = np.reshape(test_f.detach().numpy(),(ntx,nty))
    pc = ax[0,1].pcolor(X,Y,Zp, cmap=cm.coolwarm)
    pc.set_clim(vmin,vmax)

    ## covariance
    Zc = np.reshape(cov_f.detach().numpy(),(ntx,nty))
    pc = ax[0,2].pcolor(X,Y,Zc, cmap=cm.coolwarm)
    pc.set_clim(vmin,vmax)

    # plot latent outputs
    train_m = model(test_x)
    for w in range(diml):
        Zm = np.reshape(train_m[:,w].detach().numpy(),(ntx,nty))
        pc = ax[1,w].pcolor(X,Y,Zm, cmap=cm.coolwarm)
        pc.set_clim(vmin,vmax)

    ## shared colorbar
    fplot.colorbar(pc, ax=ax.ravel().tolist())

    plt.show()
