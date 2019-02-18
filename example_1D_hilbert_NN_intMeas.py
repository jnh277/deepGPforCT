import torch
import math
from matplotlib import pyplot as plt
import gp_regression_hilbert as gprh
import integration as intgr
import numpy as np
import time
import itertools as it

def cos(omega,points):
    return torch.cos(torch.squeeze(points, 1) * omega)

def cos_int(omega,points_lims):
    return torch.sin(omega*points_lims[:,1])/omega - torch.sin(omega*points_lims[:,0])/omega

def step(omega,points):
    out=points.clone()
    out[points<0.5]=-1
    out[points>0.5]=1
    return out.view(-1)

def step_int(omega,points_lims):
    out=points_lims.clone()-0.5
    out1=out[:,0].clone()
    out1[out1<0]=-out1[out[:,0]<0]
    out2=out[:,1].clone()
    out2[out[:,1]<0]=-out2[out[:,1]<0]
    return out2-out1

def stepsin(omega,points):
    step=0.5
    y = torch.sin(torch.squeeze(points,1) * omega)
    y[torch.squeeze(points, 1) >= step] += -1.0
    y[torch.squeeze(points, 1) < step] += +1.0
    return y

def stepsin_int(omega,points_lims):
    step=0.5
    a0 = torch.clamp(points_lims[:,0], max=step)
    a1 = torch.clamp(points_lims[:, 1], max=step)
    a2 = torch.clamp(points_lims[:,0], min=step)
    a3 = torch.clamp(points_lims[:, 1], min=step)
    p1 = -torch.cos(omega*a1)/omega+torch.cos(omega*a0)/omega+1.0*(a1-a0)
    p2 = -torch.cos(omega * a3) / omega + torch.cos(omega * a2) / omega - 1.0 * (a3 - a2)
    return p1+p2

integral=True
points=not(integral)

truefunc=stepsin
truefunc_int=stepsin_int
if integral:
    # INTEGRAL INPUTS
    omega=4*math.pi
    noise_std=0.001
    n = 600

    # inputs (integral limits)
    train_x=torch.rand(n,2)

    # output (integral measurements)
    train_y = truefunc_int(omega,train_x) + torch.randn(n) * noise_std

    # test points
    nt=1000
    test_x = torch.linspace(0, 1, nt).view(nt,1)
    # END INTEGRAL INPUTS

if points:
    # POINT INPUTS
    omega=4*math.pi
    noise_std=0.05
    n = 250
    train_x = torch.Tensor(n, 1)
    train_x[:, 0] = torch.linspace(0,1, n)
    train_y = truefunc(omega,train_x) + torch.randn(n) * noise_std

    nt=1000
    test_x = torch.linspace(0, 1, nt).view(nt,1)
    # END OF POINT INPUTS

# set appr params
m = [30,30] # nr of basis functions in each latent direction: Change this to add latent outputs
mt= np.prod(m) # total nr of basis functions
diml = len(m) # dimension of latent output
L = torch.empty(diml)

class DeepGP(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(DeepGP, self).__init__()
        self.linear1 = torch.nn.Linear(1, 30)
        self.tanh1 = torch.nn.Tanh()
        # self.linear2 = torch.nn.Linear(30, 30)
        # self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(30, 6)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(6, 1)
        self.tanh4 = torch.nn.Sigmoid()
        self.scale = torch.nn.Parameter(torch.Tensor([1.0]))

        if diml is 2:
            self.linear21 = torch.nn.Linear(1, 30)
            self.tanh21 = torch.nn.Tanh()
            self.linear22 = torch.nn.Linear(30, 6)
            self.tanh22 = torch.nn.Tanh()
            self.linear23 = torch.nn.Linear(6, 1)

            self.gp = gprh.GP_new(sigma_f=1, lengthscale=[1,1], sigma_n=noise_std)

        if diml is 1:
            self.gp = gprh.GP_new(sigma_f=1, lengthscale=[7], sigma_n=noise_std)

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if x_train is not None:
            h11 = x_train.clone()
            h11 = self.linear1(h11)
            h11 = self.tanh1(h11)
            # h11 = self.linear2(h11)
            # h11 = self.tanh2(h11)
            h11 = self.linear3(h11)
            h11 = self.tanh3(h11)
            h11 = self.linear4(h11)
            h11 = self.tanh4(h11)
            h11 = self.scale * h11

            if diml is 2:
                h12 = x_train.clone()
                h12 = self.linear21(h12)
                h12 = self.tanh21(h12)
                h12 = self.linear22(h12)
                h12 = self.tanh22(h12)
                h12 = self.linear23(h12)

                h = torch.cat((h11,h12),1)

            if diml is 1:
                h = h11

        if x_test is not None:
            h21 = x_test.clone()
            h21 = self.linear1(h21)
            h21 = self.tanh1(h21)
            # h21 = self.linear2(h21)
            # h21 = self.tanh2(h21)
            h21 = self.linear3(h21)
            h21 = self.tanh3(h21)
            h21 = self.linear4(h21)
            h21 = self.tanh4(h21)
            h21 = self.scale * h21

            if diml is 2:
                h22 = x_test.clone()
                h22 = self.linear21(h22)
                h22 = self.tanh21(h22)
                h22 = self.linear22(h22)
                h22 = self.tanh22(h22)
                h22 = self.linear23(h22)

                h2 = torch.cat((h21,h22),1)

            if diml is 1:
                h2 = h21
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out

model=DeepGP()

# nLL = gprh.NegMarginalLogLikelihood_deep_intMeas()  # this is the loss function
nLL = gprh.NegMarginalLogLikelihood_deep_intMeas_noBackward()  # this is the loss function

# create an index vector to store basis function permutations
index=torch.empty(mt,diml)

mmlist=[]
for q in range(diml):
    mmlist.append(np.linspace(1, m[q], m[q]))

# hard coded, but more than sufficient...
if diml is 1:
    perm = list(it.product(mmlist[0]))
elif diml is 2:
    perm = list(it.product(mmlist[0],mmlist[1]))
elif diml is 3:
    perm = list(it.product(mmlist[0],mmlist[1],mmlist[2]))
elif diml is 4:
    perm = list(it.product(mmlist[0],mmlist[1],mmlist[2],mmlist[3]))
elif diml is 5:
    perm = list(it.product(mmlist[0],mmlist[1],mmlist[2],mmlist[3],mmlist[4]))

for q in range(mt):
    index[q,:] = torch.from_numpy(np.asarray(list(it.chain.from_iterable(perm[q:q+1]))))

###### semi-vectorised integration
int_method = 2 # 1)trapezoidal, 2)simpsons standard, 3)simpsons 3/8
ni = 100 # nr of intervals
if int_method is 1:
    # trapezoidal
    sc=2*torch.ones(1,ni+1)
    sc[0,0]=1; sc[0,ni]=1
    fact = 1.0/2.0
elif int_method is 2:
    # simpsons standard
    ni = 2*round(ni/2)
    sc=torch.ones(1,ni+1)
    sc[0,ni-1]=4;
    sc[0,1:ni-1] = torch.Tensor([4,2]).repeat(1,int(ni/2-1))
    fact = 1.0/3.0
else:
    # simpsons 3/8
    ni = 3*round(ni/3)
    sc=torch.ones(1,ni+1)
    sc[0,ni-1]=3; sc[0,ni-2]=3
    sc[0,1:ni-2] = torch.Tensor([3,3,2]).repeat(1,int(ni/3-1))
    fact = 3.0/8.0

def getphi(model,L,sq_lambda):
    phi = torch.empty(n,mt)

    for q in range(n):
        a = train_x[q,0].item()
        b = train_x[q,1].item()
        h = (b-a)/ni

        zz = model(torch.linspace(a,b,ni+1).view(ni+1,1))

        intvals = torch.ones(ni+1,mt)
        for w in range(L.size(0)):
            intvals*=math.pow(L[w],-0.5)*torch.sin((zz[:,w].view(ni+1,1)+L[w])*sq_lambda[:,w].view(1,mt))

        phi[q,:] = fact*h*torch.sum(intvals*sc.t() , dim=0)
    return phi
##### semi-vectorised integration

# PICK BETTER OPTIMISER
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
],lr=0.05)

# # set numerical integration tool
# simpsons = intgr.Simpsons(fcount_out=False, fcount_max=1e3, hmin=None)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.99,min_lr=1e-6)

training_iterations = 400
tun=4
# def train():
for i in range(training_iterations):
    #def closure():
    # Zero backprop gradients
    optimizer.zero_grad()

    for q in range(diml): # todo: specify lower bounds on L (maybe we could evaluate a set of points to estimate the latent output range)
        L[q] = math.pi*m[q]*model.gp.log_lengthscale[q].exp().detach().abs()/(2.0*tun)
        sq_lambda = math.pi*index / (2.0*L)

    # t=time.time()
    if points:
        phi = torch.ones(n,mt)
        zz = model(train_x)
        for q in range(diml):
            phi *= ( 1/math.sqrt(L[q]) ) * torch.sin((zz[:,q].view(n,1)+L[q])*sq_lambda[:,q].view(1,mt)) # basis functions

    if integral:
        phi = getphi(model,L,sq_lambda)
    # print('Building phi: %.10f' %(time.time()-t))

    # Calc loss
    # t=time.time()
    loss = nLL(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi.view(n*mt,1), train_y, sq_lambda)
    # print('Computing loss: %.10f' %(time.time()-t))

    # Backprop derivatives
    # t=time.time()
    loss.backward()
    # print('Total backward: %.10f' %(time.time()-t))

    optimizer.step()
    scheduler.step(i)

    if diml is 1:
        print('Iter %d/%d - Loss: %.3f - sigf: %.3f - l: %.3f - sign: %.5f - L: %.3f - scale: %.3f' % (i + 1, training_iterations, loss.item(), model.gp.log_sigma_f.exp(), model.gp.log_lengthscale[0].exp(), model.gp.log_sigma_n.exp(), L[0], model.scale.item() ))
    elif diml is 2:
        print('Iter %d/%d - Loss: %.3f - sigf: %.3f - l1: %.3f - l2: %.3f - sign: %.5f - L1: %.3f - L2: %.3f - scale: %.3f' % (i + 1, training_iterations, loss.item(), model.gp.log_sigma_f.exp(), model.gp.log_lengthscale[0].exp(), model.gp.log_lengthscale[1].exp(), model.gp.log_sigma_n.exp(), L[0], L[1], model.scale.item() ))


# train()
# print(model)

# update L
for q in range(diml):
    L[q] = math.pi*m[q]*model.gp.log_lengthscale[q].exp().detach().abs()/(2.0*tun)

# update sq_lambda
sq_lambda = math.pi*index / (2.0*L)

# compute phi
if integral:
    phi = getphi(model,L,sq_lambda)

if points:
    phi = torch.ones(n,mt)
    zz = model(train_x)
    for q in range(diml):
        phi *= ( 1/math.sqrt(L[q]) ) * torch.sin((zz[:,q].view(n,1)+L[q])*sq_lambda[:,q].view(1,mt))

# now make predictions
test_f, cov_f = model(None, train_y, phi, sq_lambda, L, test_x)

if points:
    with torch.no_grad():
        fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

        # Plot true function as solid black
        ax.plot(test_x.numpy(), truefunc(omega,test_x).numpy(), 'k')

        # plot latent outputs
        train_m = model(test_x)
        for w in range(diml):
            ax.plot(test_x.numpy(), train_m[:,w].numpy(), 'g')

        # plot 95% credibility region
        upper = torch.squeeze(test_f, 1) + 2*cov_f.pow(0.5)
        lower = torch.squeeze(test_f, 1) - 2*cov_f.pow(0.5)
        ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        # plot predictions
        ax.plot(test_x.numpy(), test_f.detach().numpy(), 'b')

        #ax.set_ylim([-2, 2])
        ax.legend(['Observed Data', 'True', 'Predicted'])
        plt.show()

if integral:
    with torch.no_grad():
        fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot true function as solid black
        ax.plot(test_x.numpy(), truefunc(omega,test_x).numpy(), 'k')

        # # plot integral regions
        # for i in range(n):
        #     ax.plot(train_x[i,:].numpy(),np.zeros(2)-0.01*i)

        # plot latent outputs
        train_m = model(test_x)
        for w in range(diml):
            ax.plot(test_x.numpy(), train_m[:,w].numpy(), 'g')

        # plot 95% credibility region
        upper = torch.squeeze(test_f, 1) + 2*cov_f.pow(0.5)
        lower = torch.squeeze(test_f, 1) - 2*cov_f.pow(0.5)
        ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        # plot predictions
        ax.plot(test_x.numpy(), test_f.detach().numpy(), 'b')
        # ax.set_ylim([-2, 2])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        plt.show()
