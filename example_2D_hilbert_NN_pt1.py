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
    noise_std = 0.0000

    nmeas_proj = 140
    nproj = 15
    n = nmeas_proj*nproj
    nt = 150

    # import data
    dataname = 'cheese'
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
    test_x = torch.from_numpy(np.concatenate((np.reshape(X,(np.size(X),1)),np.reshape(Y,(np.size(X),1))),axis=1)).float()  # todo: circular

    # domain random points (used to limit L from below)
    ndx=100
    ndy=100
    nd=ndx*ndy
    Xd = np.linspace(-Rlim, Rlim, ndx)
    Yd = np.linspace(-Rlim, Rlim, ndy)
    Xd,Yd = np.meshgrid(Xd, Yd)
    dom_points = torch.from_numpy(np.concatenate((np.reshape(Xd,(nd,1)),np.reshape(Yd,(nd,1))),axis=1)).float()  # todo: circular

if point:
    meastype='point'

    noise_std = 0.0001
    n = 200
    nt = 200

    # import data
    dataname = 'chest'
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

######### step 1
# set appr params
m = [90,90]  # nr of basis functions in each latent direction: Change this to add latent outputs

covfunc1 = gprh.covfunc('matern',nu=2.5)
tun1 = 30

model_basic = gpnets.gpnet2_2_1(sigma_f=1, lengthscale=[1], sigma_n=1, covfunc=covfunc1) # pure GP
training_iterations_basic = 1

# loss function
lossfu_basic = gprh.NegLOOCrossValidation_phi_noBackward(model_basic.gp.covfunc)

# optimiser
optimiser_basic = FullBatchLBFGS(model_basic.parameters(), lr=1, history_size=10)

######### step 2/3
covfunc2 = gprh.covfunc('matern',nu=2.5)
tun = 30 # scaling parameter for L

model = gpnets.gpnet2_1_11(sigma_f=1,lengthscale=[1],sigma_n=1, covfunc=covfunc2) # GP/NN
m3 = [150]

######### step 2
ntp = 100 # number of training points (in each direction)
training_iterations2 = 6
optimiser2 = FullBatchLBFGS(model.parameters(), lr=1, history_size=10)
regweight2 = 0.0

######### step 3
saveFreq = 200 # how often do you wanna save the model?
training_iterations = 5

int_method = 2  # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
ni = 600  # nr of intervals in numerical integration

# loss function
lossfu = gprh.NegLOOCrossValidation_phi_noBackward(model.gp.covfunc)

# optimiser
optimiser = FullBatchLBFGS(model.parameters(), lr=1, history_size=30)
regweight = 0.0

joint = True

#########################################################
# train pure GP
#########################################################
print('\n=========Training pure GP=========')

print('Number of parameters: %d' %model_basic.npar)
dim = len(m)  # nr of latent outputs
mt = np.prod(m)  # total nr of basis functions

if integral:
    buildPhi_basic = gprh.buildPhi(m,type=meastype,tun=tun1,x0=x0,unitvecs=unitvecs,Rlim=Rlim)
else:
    buildPhi_basic = gprh.buildPhi(m,type=meastype,tun=tun1)

if integral:
    closure_basic = gprh.gp_closure(model_basic, meastype, buildPhi_basic, lossfu_basic, n, dom_points, train_y)
else:
    closure_basic = gprh.gp_closure(model_basic, meastype, buildPhi_basic, lossfu_basic, n, dom_points, train_y, train_x=train_x)
loss_basic = closure_basic()

for i in range(training_iterations_basic):

    options = {'line_search': True, 'closure': closure_basic, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser_basic.zero_grad() # zero gradients
    loss_basic.backward()  # Backprop derivatives
    loss_basic, lr, ls_step = optimiser_basic.step(options=options) # compute new loss

    # print
    gprh.optiprint(i, training_iterations_basic, loss_basic.item(), lr, ls_step, model_basic, buildPhi_basic.L)

if integral:
    test_f = gprh.compute_and_save(model_basic, meastype, dataname, train_y, n, X, Y, Z,
            ntx, nty, test_x, dom_points, m, dim, mt, noise_std, lossfu_basic, buildPhi_basic, optimiser_basic, training_iterations_basic,
             joint=True, x0=x0, unitvecs=unitvecs, Rlim=Rlim, rec_fbp=rec_fbp, err_fbp=err_fbp, basic=True)
if point:
    test_f = gprh.compute_and_save(model_basic, meastype, dataname, train_y, n, X, Y, Z,
        ntx, nty, test_x, dom_points, m, dim, mt, noise_std, lossfu_basic, buildPhi_basic, optimiser_basic, training_iterations_basic,
         joint=True, train_x=train_x, basic=True)

try:  # since plotting might produce an error on remote machines
    vmin = 0
    vmax = Z.max()
    gprh.makeplot2D_new('mymodel_basic_'+meastype+'_'+dataname+'_'+str(training_iterations_basic),vmin=vmin,vmax=vmax,data=True)
except:
    pass


#########################################################
# pre-training the network
#########################################################
print('\n=========Pre-training neural network=========')

print('Number of parameters: %d' %model.npar)

nextract = np.int(np.ceil(nt/ntp/ntp))

test_f2 = test_f.detach()[0::nextract]
test_x2 = test_x[0::nextract,:]

def closure2():
    if regweight2==0:
        return (model(test_x2) .sub( test_f2 ) ).pow(2) .sum() .div( test_f2.numel() )
    else:
        return (model(test_x2) .sub( test_f2 ) ).pow(2) .sum() .div( test_f2.numel() ) .add( gprh.regulariser(model,weight=regweight2) )

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
# joint training
#########################################################
print('\n=========Training the joint model=========')

# set appr params
m = m3  # nr of basis functions in each latent direction: Change this to add latent outputs
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
        vmax = 1.2
        gprh.makeplot2D_new('mymodel_'+meastype+'_'+dataname+'_'+str(training_iterations),vmin=vmin,vmax=vmax,data=True)
    except:
        pass

else:
    gprh.compute_and_save(0)
