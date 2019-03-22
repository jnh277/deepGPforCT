import torch
import gp_regression_hilbert as gprh
import gpnets
import numpy as np
import radon_data as rd
import math

# add path to optimisation code
import sys
sys.path.append('./opti_functions/')
from Adam_ls_alt import Adam_ls_alt
from Adam_ls import Adam_ls
from LBFGS import FullBatchLBFGS

# use integral or point measurements
integral = True
points = not(integral)

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

if points:
    meastype='point'

    # select true function
    truefunc=gprh.circlefunc2

    noise_std=0.001
    n = 2000

    # input data
    train_x = torch.rand(n,2)
    train_y = truefunc(train_x) + torch.randn(n) * noise_std

    # test point
    ntx=200
    nty=200
    nt=ntx*nty
    X = np.linspace(0, 1, ntx)
    Y = np.linspace(0, 1, nty)
    X,Y = np.meshgrid(X, Y)
    test_x = torch.from_numpy(np.concatenate((np.reshape(X,(nt,1)),np.reshape(Y,(nt,1))),axis=1)).float()

    # domain random points (used to limit L from below)
    ndx=30
    ndy=30
    nd=ndx*ndy
    Xd = np.linspace(-1, 1, ndx)
    Yd = np.linspace(-1, 1, ndy)
    Xd,Yd = np.meshgrid(Xd, Yd)
    dom_points = torch.from_numpy(np.concatenate((np.reshape(Xd,(nd,1)),np.reshape(Yd,(nd,1))),axis=1)).float()

# set appr params
m = [60] # nr of basis functions in each latent direction: Change this to add latent outputs
diml=len(m) # nr of latent outputs
mt= np.prod(m) # total nr of basis functions

# select model
model = gpnets.gpnet2_1_4(sigma_f=1,lengthscale=[1],sigma_n=0.05)
if not(hasattr(model, 'pureGP')):
    setattr(model, 'pureGP', False)

# loss function
nLL = gprh.NegMarginalLogLikelihood_phi_noBackward()

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

    # plot
    if points:
        gprh.makeplot2D(model,X,Y,ntx,nty,test_f,cov_f,diml,test_x=test_x,truefunc=truefunc,train_x=train_x,type=meastype,vmin=-2,vmax=2)

    # RMS error
    if integral:
        ground_truth = torch.from_numpy(Z).float().view(np.size(Z))
    else:
        ground_truth = truefunc(test_x)
    error = torch.mean( (ground_truth - test_f.squeeze()).pow(2) ).sqrt()
    print('RMS error: %.10f' %(error.item()))
    if integral:
        print('RMS error fbp: %.10f' %(err_fbp))

    if integral:
        # save variables
        torch.save((model, dataname, train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp,
                    ntx, nty, test_x, dom_points, m, diml, mt,
                    test_f, cov_f, noise_std, nLL, buildPhi, optimiser.param_groups[0]['lr'], it_number),
                   'mymodel_'+dataname+'_'+str(it_number))

# optimiser
optimiser = Adam_ls(model.parameters(), lr=0.01, weight_decay=0.0)  # Adam with line search

# compute initial loss
def closure():
    global L
    if integral:
        phi, sq_lambda, L = buildPhi.getphi(model,m,n,mt,dom_points)
    else:
        phi, sq_lambda, L = buildPhi.getphi(model,m,n,mt,dom_points,train_x=train_x)
    loss = nLL(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi, train_y, sq_lambda)
    return loss
loss = closure()

saveFreq=3 # how often do you wanna save the model?

training_iterations = 5000
for i in range(training_iterations):
    # build phi

    options = {'line_search': True, 'closure': closure, 'max_ls': 5, 'ls_debug': False, 'inplace': False, 'interpolate': False,
               'eta': 3, 'c1': 1e-6, 'decrease_lr_on_max_ls': 0.5, 'increase_lr_on_min_ls': 2}

    optimiser.zero_grad()  # zero gradients
    loss.backward()  # Backprop derivatives
    loss, lr, ls_step = optimiser.step(options=options) # compute new loss

    # print
    gprh.optiprint(i, training_iterations,loss.item(), lr, ls_step,model, L)

    if integral:
        if (i+1)%saveFreq==0:
            compute_and_save(i+1)


compute_and_save(training_iterations)

try: # since plotting might produce an error on remote machines
    vmin = -2
    vmax = 2
    gprh.makeplot2D_new('mymodel_'+dataname+'_'+str(training_iterations),vmin=vmin,vmax=vmax)
except:
    pass
