import torch
import gp_regression_hilbert as gprh
import gpnets
import numpy as np
import radon_data as rd
import math
from time import time

# add path to optimisation code
import sys
sys.path.append('./opti_functions/')
from Adam_ls_alt import Adam_ls_alt
from Adam_ls import Adam_ls
from LBFGS import FullBatchLBFGS
from LBFGS_alt import FullBatchLBFGS as FullBatchLBFGS_alt

# use integral or point measurements
integral = False
point = not(integral)

if integral:
    meastype='int'

    # noise level
    noise_std = 0.001

    # import data
    dataname='phantom'
    print('Getting data...')
    train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp = rd.getdata(dataname=dataname,image_res=3e3,nmeas_proj=100,
                                                                           nproj=20,nt=None,sigma_n=noise_std,reconstruct_fbp=True)
    print('Data ready!')

    # convert
    train_y = torch.from_numpy(train_y).float()
    x0 = torch.from_numpy(x0).float()
    unitvecs = torch.from_numpy(unitvecs).float()

    # test points
    ntx=np.size(X,1)
    nty=np.size(X,0)
    test_x = torch.from_numpy(np.concatenate((np.reshape(X,(np.size(X),1)),np.reshape(Y,(np.size(X),1))),axis=1)).float()

    # domain random points (used to limit L from below)
    ndx=30
    ndy=30
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
    dataname='phantom'
    print('Getting data...')
    train_y, train_x, X, Y, Z = rd.getdata(dataname=dataname,image_res=3e3,nt=np.int(np.sqrt(6*n)),sigma_n=noise_std,points=True,npmeas=n)
    print('Data ready!')

    # convert
    train_y = torch.from_numpy(train_y).float()
    train_x = torch.from_numpy(train_x).float()

    # train_y = train_y-train_y.mean()

    # test points
    ntx=np.size(X,1)
    nty=np.size(X,0)
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
    ndx=30
    ndy=30
    nd=ndx*ndy
    Xd = np.linspace(-1, 1, ndx)
    Yd = np.linspace(-1, 1, ndy)
    Xd,Yd = np.meshgrid(Xd, Yd)
    dom_points = torch.from_numpy(np.concatenate((np.reshape(Xd,(nd,1)),np.reshape(Yd,(nd,1))),axis=1)).float()

# set appr params
m = [60]  # nr of basis functions in each latent direction: Change this to add latent outputs
diml = len(m)  # nr of latent outputs
mt = np.prod(m)  # total nr of basis functions

# select model
model = gpnets.gpnet2_1_2(sigma_f=1, lengthscale=[1,1], sigma_n=1)
print('Number of parameters: %d' %model.npar)

# loss function
lossfu = gprh.NegLOOCrossValidation_phi_noBackward()
# lossfu = gprh.NegMarginalLogLikelihood_phi_noBackward()

# buildPhi object
tun=4 # scaling parameter for L (nr of "std":s)
if integral:
    int_method = 2  # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
    ni = 200  # nr of intervals in numerical integration
    buildPhi = gprh.buildPhi(m,type=meastype,ni=ni,int_method=int_method,tun=tun,x0=x0,unitvecs=unitvecs,Rlim=Rlim)
else:
    buildPhi = gprh.buildPhi(m,type=meastype,tun=tun)

# function that computes the solution and save stuff (declared here to enable usage within the optimisation loop)
def compute_and_save(it_number):
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

# optimiser
optimiser = FullBatchLBFGS(model.parameters(), lr=1, history_size=30)

# compute initial loss
def closure():
    optimiser.zero_grad()
    global L
    if integral:
        phi, sq_lambda, L = buildPhi.getphi(model,m,n,mt,dom_points)
    else:
        phi, sq_lambda, L = buildPhi.getphi(model,m,n,mt,dom_points,train_x=train_x)
    return lossfu(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi, train_y, sq_lambda)
loss = closure()

saveFreq = 100 # how often do you wanna save the model?

training_iterations = 500

tg_tot = 0
topt_tot = 0
for i in range(training_iterations):

    options = {'closure': closure, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

    optimiser.zero_grad() # zero gradients
    tg=time()
    loss.backward()  # Backprop derivatives
    tg_tot+=time()-tg
    topt=time()
    loss, lr, ls_step = optimiser.step(options=options) # compute new loss
    topt_tot+=time()-topt

    # print
    gprh.optiprint(i, training_iterations, loss.item(), lr, ls_step, model, L)

    if (i+1)%saveFreq==0:
        compute_and_save(i+1)


compute_and_save(training_iterations)

try:  # since plotting might produce an error on remote machines
    vmin = 0
    vmax = 1
    gprh.makeplot2D_new('mymodel_'+meastype+'_'+dataname+'_'+str(training_iterations),vmin=vmin,vmax=vmax)
except:
    pass

print(tg_tot/topt_tot)  # how big part is gradient computations?
