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
    noise_std = 0.0001

    nmeas_proj = 20
    nproj = 30
    n = nmeas_proj*nproj
    nt = 50

    # import data
    dataname = 'circle_square'
    print('Getting data...')
    train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp = rd.getdata(dataname=dataname,R=1,image_res=3e3,nmeas_proj=nmeas_proj,
                                                                           nproj=nproj,nt=nt,
                                                                           sigma_n=noise_std,reconstruct_fbp=True)
    # (train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp) = torch.load(dataname+'_tostart')
    print('Data ready!')
    # torch.save((train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp),dataname+'_tostart')

    # convert
    train_y = torch.from_numpy(train_y).float()
    x0 = torch.from_numpy(x0).float()
    unitvecs = torch.from_numpy(unitvecs).float()

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
    dom_points = dom_points[dom_points.pow(2).sum(dim=1).sqrt() < Rlim, :]

if point:
    meastype='point'

    noise_std = 0.0001
    n = 2000
    nt = 100

    # import data
    dataname = 'cheese'
    print('Getting data...')
    train_y, train_x, X, Y, Z = rd.getdata(dataname=dataname,image_res=3e3,nt=nt,
                                           sigma_n=noise_std,points=True,npmeas=n)
    print('Data ready!')

    # convert
    train_y = torch.from_numpy(train_y).float()
    train_x = torch.from_numpy(train_x).float()

    # test points
    ntx = np.size(X,1)
    nty = np.size(X,0)
    test_x = torch.from_numpy(np.concatenate((np.reshape(X,(np.size(X),1)),np.reshape(Y,(np.size(X),1))),axis=1)).float()

    # domain random points (used to limit L from below)
    ndx = 100
    ndy = 100
    nd = ndx*ndy
    Xd = np.linspace(-1, 1, ndx)
    Yd = np.linspace(-1, 1, ndy)
    Xd,Yd = np.meshgrid(Xd, Yd)
    dom_points = torch.from_numpy(np.concatenate((np.reshape(Xd,(nd,1)),np.reshape(Yd,(nd,1))),axis=1)).float()

######################## details #########################

# define model
# model = gpnets.gpnet2_2_2(sigma_f=1,lengthscale=[1,1],sigma_n=1, covfunc=covfunc2) # GP/NN
model = gpnets.gpnet2_1_11(sigma_f=1,lengthscale=[1],sigma_n=1, covfunc= gprh.covfunc('matern',nu=2.5)) # GP/NN

######### pre-train network
training_iterations_pt = 2000
optimiser_pt = FullBatchLBFGS(model.parameters(), lr=1, history_size=10)
regweight_pt = 0.0

int_method_pt = 2  # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
ni_pt = 500  # nr of intervals in numerical integration

######### joint training
saveFreq = 2000 # how often do you wanna save the model?
training_iterations = 20

int_method = 2  # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
ni = 500  # nr of intervals in numerical integration

m = [150]
tun = 30 # scaling parameter for L

# loss function
lossfu = gprh.NegLOOCrossValidation_phi_noBackward(model.gp.covfunc)
# lossfu = gprh.NegMarginalLogLikelihood_phi_noBackward(model.gp.covfunc)

# optimiser
optimiser = FullBatchLBFGS(model.parameters(), lr=1, history_size=10)
regweight = 0.0

joint = True


#########################################################
# pre-train network
#########################################################
print('\n=========Pre-training neural network=========')

if integral:
    closure2 = gprh.net_closure(model,meastype,train_y,ni=ni_pt,int_method=int_method_pt,x0=x0,unitvecs=unitvecs,Rlim=Rlim,regweight=regweight_pt)
else:
    closure2 = gprh.net_closure(model,meastype,train_y,train_x=train_x,regweight=regweight_pt)

loss2 = closure2()

for i in range(training_iterations_pt):
    options = {'line_search': True, 'closure': closure2, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser_pt.zero_grad() # zero gradients
    loss2.backward() # propagate derivatives
    loss2, lr, ls_step = optimiser_pt.step(options=options) # compute new loss

    # print
    gprh.optiprint(i, training_iterations_pt, loss2.item(), lr, ls_step)


#########################################################
# joint training
#########################################################
print('\n=========Training the joint model=========')

# set appr params
dim = len(m)  # nr of latent outputs
mt = np.prod(m)  # total nr of basis functions

# buildPhi object
if integral:
    buildPhi = gprh.buildPhi(m,type=meastype,ni=ni,int_method=int_method,tun=tun,x0=x0,unitvecs=unitvecs,Rlim=Rlim)
else:
    buildPhi = gprh.buildPhi(m,type=meastype,tun=tun)


if joint:

    if integral:
        closure = gprh.gp_closure(model, meastype, buildPhi, lossfu, n, dom_points, train_y, regweight=regweight)
    else:
        closure = gprh.gp_closure(model, meastype, buildPhi, lossfu, n, dom_points, train_y, train_x=train_x, regweight=regweight)

    loss = closure()

    for i in range(training_iterations):

        options = {'line_search': True, 'closure': closure, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
                   'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

        optimiser.zero_grad() # zero gradients
        loss.backward()  # Backprop derivatives
        loss, lr, ls_step = optimiser.step(options=options) # compute new loss

        # print
        gprh.optiprint(i, training_iterations, loss.item(), lr, ls_step, model, buildPhi.L)

        if i%saveFreq==0:
            if integral:
                gprh.compute_and_save(model, meastype, dataname, train_y, n, X, Y, Z,
                        ntx, nty, test_x, dom_points, m, dim, mt, noise_std, lossfu, buildPhi, optimiser, i,
                         joint, x0=x0, unitvecs=unitvecs, Rlim=Rlim, rec_fbp=rec_fbp, err_fbp=err_fbp)
            if point:
                gprh.compute_and_save(model, meastype, dataname, train_y, n, X, Y, Z,
                    ntx, nty, test_x, dom_points, m, dim, mt, noise_std, lossfu, buildPhi, optimiser, i,
                     joint, train_x=train_x)

    if integral:
        gprh.compute_and_save(model, meastype, dataname, train_y, n, X, Y, Z,
                ntx, nty, test_x, dom_points, m, dim, mt, noise_std, lossfu, buildPhi, optimiser, training_iterations,
                 joint, x0=x0, unitvecs=unitvecs, Rlim=Rlim, rec_fbp=rec_fbp, err_fbp=err_fbp)
    if point:
        gprh.compute_and_save(model, meastype, dataname, train_y, n, X, Y, Z,
            ntx, nty, test_x, dom_points, m, dim, mt, noise_std, lossfu, buildPhi, optimiser, training_iterations,
             joint, train_x=train_x)

    try:  # since plotting might produce an error on remote machines
        vmin = -0.5
        vmax = 3
        gprh.makeplot2D_new('mymodel_'+meastype+'_'+dataname+'_'+str(training_iterations),vmin=vmin,vmax=vmax,data=True)
    except:
        pass

else:
    gprh.compute_and_save(0)
