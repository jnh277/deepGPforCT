import torch
import gp_regression_hilbert as gprh
import gpnets
import numpy as np
import radon_data as rd

# use integral or point measurements
integral=True
points=not(integral)

if integral:
    meastype='int'

    # noise level
    noise_std = 0.005

    # import data
    dataname='circle_square'
    print('Getting data...')
    train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp = rd.getdata(dataname=dataname,nmeas_proj=100,nproj=20,nt=None,sigma_n=noise_std,reconstruct_fbp=True)
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
    truefunc=gprh.circlefunc

    noise_std=0.01
    n = 1500

    # input data
    train_x = -1+2*torch.rand(n,2)
    train_y = truefunc(train_x) + torch.randn(n) * noise_std

    # test point
    ntx=200
    nty=200
    nt=ntx*nty
    X = np.linspace(-1, 1, ntx)
    Y = np.linspace(-1, 1, nty)
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
m = [50] # nr of basis functions in each latent direction: Change this to add latent outputs
diml=len(m) # nr of latent outputs
mt= np.prod(m) # total nr of basis functions

# select model
model = gpnets.gpnet2_1_4()
# model=torch.load("mymodel")

# loss function
nLL = gprh.NegMarginalLogLikelihood_phi_noBackward()

# buildPhi object
tun=4 # scaling parameter for L (nr of "std":s)
if integral:
    int_method=2 # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
    ni=200 # nr of intervals in numerical integration
    buildPhi = gprh.buildPhi(m,type=meastype,ni=ni,int_method=int_method,tun=tun,x0=x0,unitvecs=unitvecs,Rlim=Rlim)
else:
    buildPhi = gprh.buildPhi(m,type=meastype,tun=tun)

# optimiser
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
],lr=0.05)

# scheduler for optimisation
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.99,min_lr=1e-6)

saveFreq=1000 # how often do you wanna save the model?

training_iterations = 10000
for i in range(training_iterations):
    # Zero backprop gradients
    optimizer.zero_grad()

    # build phi
    if integral:
        phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,dom_points)
    else:
        phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,dom_points,train_x=train_x)

    # Calc loss
    loss = nLL(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi, train_y, sq_lambda)

    # Backprop derivatives
    loss.backward()

    # step forward
    optimizer.step()
    scheduler.step(i)

    # print
    gprh.optiprint(i,training_iterations,loss.item(),model,L)

    if integral:
        if (i+1)%saveFreq==0:
            torch.save(model,'mymodel_'+dataname+'_'+str(i+1))

# update phi
if integral:
    phi = buildPhi.getphi(model,m,n,mt,dom_points)[0]
else:
    phi = buildPhi.getphi(model,m,n,mt,dom_points,train_x=train_x)[0]

# now make predictions
test_f, cov_f = model(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)

# plot
if integral:
    gprh.makeplot2D(model,X,Y,ntx,nty,test_f,cov_f,diml,test_x=test_x,Z=Z,type=meastype,vmin=-2,vmax=2)
else:
    gprh.makeplot2D(model,X,Y,ntx,nty,test_f,cov_f,diml,test_x=test_x,truefunc=truefunc,train_x=train_x,type=meastype,vmin=-2,vmax=2)

# L2 norm
if integral:
    ground_truth = torch.from_numpy(Z).float().view(np.size(Z))
else:
    ground_truth = truefunc(test_x)
error = torch.mean( (ground_truth - test_f.squeeze()).pow(2) ).sqrt()
print('L2 error norm: %.10f' %(error.item()))

# final save
torch.save((model,n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp, test_f,cov_f, test_x),'mymodel_'+dataname+'_'+str(training_iterations)+'_all')
