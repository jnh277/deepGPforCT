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
integral = True
point = not(integral)

if integral:
    meastype='int'

    # noise level
    noise_std = 0.00001

    nmeas_proj = 185
    nproj = 9
    n = nmeas_proj*nproj

    # import data
    dataname = 'phantom'
    print('Getting data...')
    train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp = rd.getdata(dataname=dataname,image_res=3e3,nmeas_proj=nmeas_proj,
                                                                           nproj=nproj,nt=80,
                                                                           sigma_n=noise_std,reconstruct_fbp=True)
    # (train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp) = torch.load(dataname+'_tostart')
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

    noise_std = 0.0001
    n = 600

    # import data
    dataname = 'circle_square'
    print('Getting data...')
    train_y, train_x, X, Y, Z = rd.getdata(dataname=dataname,image_res=3e3,nt=80,
                                           sigma_n=noise_std,points=True,npmeas=n)
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

######################## details #########################
# set appr params
m = [70,70]  # nr of basis functions in each latent direction: Change this to add latent outputs
diml = len(m)  # nr of latent outputs
mt = np.prod(m)  # total nr of basis functions

######### step 1
model_basic = gpnets.gpnet2_2_1(sigma_f=1, lengthscale=[1,1], sigma_n=1) # pure GP
# model_basic.pureGP=False
saveFreq_basic = 2000 # how often do you wanna save the model?
training_iterations_basic = 50

# loss function
lossfu_basic = gprh.NegLOOCrossValidation_phi_noBackward()

# optimiser
optimiser_basic = FullBatchLBFGS(model_basic.parameters(), lr=1, history_size=30)

######### step 2/3
# model = gpnets.gpnet2_2_3(sigma_f=1,lengthscale=[1,1],sigma_n=1) # GP/NN
# m3 = [50,50]

model = gpnets.gpnet2_1_10(sigma_f=1,lengthscale=[1],sigma_n=1) # GP/NN
m3 = [150]

######### step 2
training_iterations2 = 4000
optimiser2 = FullBatchLBFGS(model.parameters(), lr=1, history_size=30)

######### step 3
saveFreq = 4400 # how often do you wanna save the model?
training_iterations = 600

# loss function
lossfu = gprh.NegLOOCrossValidation_phi_noBackward()

# optimiser
optimiser = FullBatchLBFGS(model.parameters(), lr=1, history_size=30)

step_3 = False

#########################################################
# STEP 1
#########################################################
print('Number of parameters: %d' %model_basic.npar)

# buildPhi object
tun=4 # scaling parameter for L (nr of "std":s)
if integral:
    int_method = 2  # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
    ni = 200  # nr of intervals in numerical integration
    buildPhi_basic = gprh.buildPhi(m,type=meastype,ni=ni,int_method=int_method,tun=tun,x0=x0,unitvecs=unitvecs,Rlim=Rlim)
else:
    buildPhi_basic = gprh.buildPhi(m,type=meastype,tun=tun)

# compute initial loss
def closure_basic():
    optimiser_basic.zero_grad()
    global L
    if integral:
        phi, sq_lambda, L = buildPhi_basic.getphi(model_basic,m,n,mt,dom_points)
    else:
        phi, sq_lambda, L = buildPhi_basic.getphi(model_basic,m,n,mt,dom_points,train_x=train_x)
    return lossfu_basic(model_basic.gp.log_sigma_f, model_basic.gp.log_lengthscale, model_basic.gp.log_sigma_n, phi, train_y, sq_lambda)
loss_basic = closure_basic()

# function that computes the solution and save stuff (declared here to enable usage within the optimisation loop)
def compute_and_save_basic(it_number):
    # update phi
    if integral:
        phi,sq_lambda,L = buildPhi_basic.getphi(model_basic,m,n,mt,dom_points)
    else:
        phi,sq_lambda,L = buildPhi_basic.getphi(model_basic,m,n,mt,dom_points,train_x=train_x)

    # now make predictions
    test_f, cov_f = model_basic(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)

    # save variables
    if integral:
        torch.save((model_basic, dataname, train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp,
                    ntx, nty, test_x, dom_points, m, diml, mt,
                    test_f, cov_f, noise_std, lossfu_basic, buildPhi_basic, optimiser_basic.__getstate__(), it_number),
                   'mymodel_basic_'+meastype+'_'+dataname+'_'+str(it_number))
    if point:
        torch.save((model_basic, dataname, train_y, n, train_x, X, Y, Z,
                    ntx, nty, test_x, dom_points, m, diml, mt,
                    test_f, cov_f, noise_std, lossfu_basic, buildPhi_basic, optimiser_basic.__getstate__(), it_number),
                   'mymodel_basic_'+meastype+'_'+dataname+'_'+str(it_number))

    return test_f, cov_f

for i in range(training_iterations_basic):

    options = {'line_search': True, 'closure': closure_basic, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser_basic.zero_grad() # zero gradients

    loss_basic.backward()  # Backprop derivatives

    loss_basic, lr, ls_step = optimiser_basic.step(options=options) # compute new loss

    # print
    gprh.optiprint(i, training_iterations_basic, loss_basic.item(), lr, ls_step, model_basic, L)

test_f, cov_f = compute_and_save_basic(training_iterations_basic)

try:  # since plotting might produce an error on remote machines
    vmin = 0
    vmax = 1
    gprh.makeplot2D_new('mymodel_basic_'+meastype+'_'+dataname+'_'+str(training_iterations_basic),vmin=vmin,vmax=vmax,data=True)
except:
    pass


#########################################################
# STEP 2
#########################################################
print('Number of parameters: %d' %model.npar)

test_f = test_f.detach()

def closure2():
    optimiser2.zero_grad() # zero gradients
    return (model(test_x) .sub( test_f ) ).pow(2) .sum()

loss2 = closure2()

for i in range(training_iterations2):
    options = {'line_search': True, 'closure': closure2, 'max_ls': 5, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser2.zero_grad() # zero gradients
    loss2.backward() # propagate derivatives
    loss2, lr, ls_iters = optimiser2.step(options=options) # compute new loss

    # print
    print('Iter %d/%d - Loss: %.5f - lr: %.5f - LS iters: %0.0f' %(i,training_iterations2,loss2.item(),lr,ls_iters))


#########################################################
# STEP 3
#########################################################
# set appr params
m = m3  # nr of basis functions in each latent direction: Change this to add latent outputs
diml = len(m)  # nr of latent outputs
mt = np.prod(m)  # total nr of basis functions

# buildPhi object
tun=4 # scaling parameter for L (nr of "std":s)
if integral:
    int_method = 2  # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
    ni = 200  # nr of intervals in numerical integration
    buildPhi = gprh.buildPhi(m,type=meastype,ni=ni,int_method=int_method,tun=tun,x0=x0,unitvecs=unitvecs,Rlim=Rlim)
else:
    buildPhi = gprh.buildPhi(m,type=meastype,tun=tun)


# regularizer todo: build this into the optimiser
def regularizer(model):
    reg = torch.zeros(1)
    for p in model.parameters():
        reg = reg.add(  p.pow(2).sum()  )
    return reg


def compute_and_save(it_number):
    if step_3:
        # update phi
        if integral:
            phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,dom_points)
        else:
            phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,dom_points,train_x=train_x)

        # now make predictions
        test_f, cov_f = model(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)

        # RMS error
        ground_truth = torch.from_numpy(Z).float().view(np.size(Z))
        error = torch.mean( (ground_truth - test_f.squeeze()).pow(2) ).sqrt()
        print('RMS error: %.10f' %(error.item()))
        if integral:
            print('RMS error fbp: %.10f' %(err_fbp))
    else:
        test_f = None
        cov_f = None

    # save variables
    if integral:
        torch.save((model, dataname, train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp,
                    ntx, nty, test_x, dom_points, m, diml, mt,
                    test_f, cov_f, noise_std, lossfu, buildPhi, optimiser.__getstate__(), it_number),
                   'mymodel_'+meastype+'_'+dataname+'_'+str(it_number))
    if point:
        torch.save((model, dataname, train_y, n, train_x, X, Y, Z,
                    ntx, nty, test_x, dom_points, m, diml, mt,
                    test_f, cov_f, noise_std, lossfu, buildPhi, optimiser.__getstate__(), it_number),
                   'mymodel_'+meastype+'_'+dataname+'_'+str(it_number))


if step_3:
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

    for i in range(training_iterations):

        options = {'line_search': True, 'closure': closure, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
                   'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

        optimiser.zero_grad() # zero gradients

        loss.backward()  # Backprop derivatives

        loss, lr, ls_step = optimiser.step(options=options) # compute new loss

        # print
        gprh.optiprint(i, training_iterations, loss.item(), lr, ls_step, model, L)

        if (i+1)%saveFreq==0:
            compute_and_save(i+1)


    compute_and_save(training_iterations)

    try:  # since plotting might produce an error on remote machines
        vmin = 0
        vmax = 1
        gprh.makeplot2D_new('mymodel_'+meastype+'_'+dataname+'_'+str(training_iterations),vmin=vmin,vmax=vmax,data=True)
    except:
        pass
else:
    compute_and_save(0)
