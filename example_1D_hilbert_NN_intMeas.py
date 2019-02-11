import torch
import math
from matplotlib import pyplot as plt
import gp_regression_hilbert as gprh
import integration as intgr
import numpy as np
import time

def cosfunc(omega,points):
    return torch.cos(torch.squeeze(points, 1) * omega)

def cosfunc_int(omega,points_lims):
    return torch.sin(omega*points_lims[:,1])/omega - torch.sin(omega*points_lims[:,0])/omega

def stepfunc(points):
    out=points.clone()
    out[points<0.5]=-1
    out[points>0.5]=1
    return out.view(-1)

def stepfunc_int(points_lims):
    out=points_lims.clone()-0.5
    out1=out[:,0].clone()
    out1[out1<0]=-out1[out[:,0]<0]
    out2=out[:,1].clone()
    out2[out[:,1]<0]=-out2[out[:,1]<0]
    return out2-out1

integral=True
points=False#True

if integral:
    # INTEGRAL INPUTS
    omega=4*math.pi
    noise_std=0.001
    n = 50

    # inputs (integral limits)
    train_x=torch.rand(n,2)

    # output (integral measurements)
    train_y = stepfunc_int(train_x) + torch.randn(n) * noise_std
    # train_y = cosfunc_int(omega,train_x) + torch.randn(n) * noise_std

    # test points
    nt=1000
    test_x = torch.linspace(0, 1, nt).view(nt,1)
    # END INTEGRAL INPUTS

if points:
    # POINT INPUTS
    omega=4*math.pi
    noise_std=0.05
    n = 50
    train_x = torch.Tensor(n, 1)
    train_x[:, 0] = torch.linspace(0,1, n)
    train_y = stepfunc(train_x) + torch.randn(n) * noise_std
    # train_y = cosfunc(omega,train_x) + torch.randn(n) * noise_std

    nt=1000
    test_x = torch.linspace(0, 1, nt).view(nt,1)
    # END OF POINT INPUTS

# set appr params
m = 10 # nr of basis functions

class DeepGP(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(DeepGP, self).__init__()
        self.linear1 = torch.nn.Linear(1, 30)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(30, 30)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(30, 6)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(6, 1)
        self.tanh4 = torch.nn.Sigmoid()
        self.gp = gprh.GP_1D_new(sigma_f=1, lengthscale=30, sigma_n=noise_std)

    def forward(self, x_train=None, y_train=None, phi=None, m=None, L=None, x_test=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if x_train is not None:
            h = x_train.clone()

            h = self.linear1(h)
            h = self.tanh1(h)
            h = self.linear2(h)
            h = self.tanh2(h)
            h = self.linear3(h)
            h = self.tanh3(h)
            h = self.linear4(h)
            h = self.tanh4(h)

            # h[x_train<0.5]=-1
            # h[x_train>0.5]=1
        if x_test is not None:
            h2 = x_test.clone()

            h2 = self.linear1(h2)
            h2 = self.tanh1(h2)
            h2 = self.linear2(h2)
            h2 = self.tanh2(h2)
            h2 = self.linear3(h2)
            h2 = self.tanh3(h2)
            h2 = self.linear4(h2)
            h2 = self.tanh4(h2)

            # h2[x_test<0.5]=-1
            # h2[x_test>0.5]=1
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,m,L,h2)
        else:
            out = h
        return out

# model=torch.load("mymodel")

model=DeepGP()

model.gp.covtype="se"
nLL = gprh.NegMarginalLogLikelihood_deep_intMeas(model.gp.covtype,model.gp.nu)  # this is the loss function

# create an index vector, index=[1 2 3...]
index = torch.linspace(1, m, m).view(1,m)

###### semi-vectorised integration
def getphi(model,L):
    phi = torch.Tensor(n,m)
    omega = math.pi*index / (2.0*L)

    ni = 500
    ni = 3*round(ni/3)
    sc=torch.ones(1,ni+1)
    sc[0,ni-1]=3; sc[0,ni-2]=3
    sc[0,1:ni-2] = torch.Tensor([3,3,2]).repeat(1,int(ni/3-1))
    for q in range(n):
        a = train_x[q,0].item()
        b = train_x[q,1].item()
        h = (b-a)/ni

        points2eval = torch.linspace(a,b,ni+1).view(ni+1,1)

        phi[q,:] = (3*h/8)*math.pow(L,-0.5)*torch.sum(torch.sin((model(points2eval)+L)*omega)*sc.t() , dim=0)
    return phi
##### semi-vectorised integration

# PICK BETTER OPTIMISER
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
],lr=0.1)

# # set numerical integration tool
# simpsons = intgr.Simpsons(fcount_out=False, fcount_max=1e3, hmin=None)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.95,min_lr=1e-6)

training_iterations = 400
tun=3
# def train():
for i in range(training_iterations):
    #def closure():
    # Zero backprop gradients
    optimizer.zero_grad()

    L = max(1.5,math.pi*m*torch.sqrt(model.gp.log_lengthscale.exp().detach().pow(2))/(2.0*tun))

    if points:
        phi = ( 1/math.sqrt(L) ) * torch.sin(math.pi*index*(model(train_x)+L)*0.5/L) # basis functions

    if integral:
        # t=time.time()
        # # calculate phi using numerical integration
        # phi = torch.zeros(n,m)
        # for k in range(n):
        #     for q in range(m):
        #         phi[k,q] = simpsons(lambda x: ( 1/math.sqrt(L) )*torch.sin(math.pi*(q+1)*(model(x.view(1))+L)*0.5/L), train_x[k,0].view(1), train_x[k,1].view(1), 1e-6)
        # print(time.time()-t)

        # t=time.time()
        # # 1 for-loop
        # phi2 = torch.zeros(n*m,1)
        # for k in range(n*m):
        #     j=math.floor(k/m)
        #     q=k%m
        #     phi2[k,0] = simpsons(lambda x: ( 1/math.sqrt(L) )*torch.sin(math.pi*(q+1)*(model(x.view(1))+L)*0.5/L), train_x[j,0].view(1), train_x[j,1].view(1), 1e-6)
        # print(time.time()-t)

        # t=time.time()
        phi = getphi(model,L)
        # print(time.time()-t)

    # Calc loss and backprop derivatives
    loss = nLL(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi.view(n*m,1), train_y, m, L)
    loss.backward()

    optimizer.step()
    scheduler.step(i)

    print('Iter %d/%d - Loss: %.3f - sigf: %.3f - l: %.3f - sign: %.3f - L: %.3f' % (i + 1, training_iterations, loss.item(), model.gp.log_sigma_f.exp(), model.gp.log_lengthscale.exp(), model.gp.log_sigma_n.exp() ,L ))


# train()
print(model)

L = max(1.5,math.pi*m*torch.sqrt(model.gp.log_lengthscale.exp().detach().pow(2))/(2.0*tun))
if integral:
    phi = getphi(model,L)

if points:
    phi = ( 1/math.sqrt(L) ) * torch.sin(math.pi*index*(model(train_x)+L)*0.5/L) # basis functions

# now make predictions
test_f, cov_f = model(None, train_y, phi, m, L, test_x)

if points:
    with torch.no_grad():
        fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

        # Plot true function as solid black
        ax.plot(test_x.numpy(), stepfunc(test_x).numpy(), 'k')
        # ax.plot(test_x.numpy(), cosfunc(omega,test_x).numpy(), 'k')

        # plot h
        train_m = model(test_x)
        ax.plot(test_x.numpy(), train_m.numpy(), 'g')

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
        ax.plot(test_x.numpy(), stepfunc(test_x).numpy(), 'k')
        # ax.plot(test_x.numpy(), cosfunc(omega,test_x).numpy(), 'k')

        # plot integral regions
        for i in range(n):
            ax.plot(train_x[i,:].numpy(),np.zeros(2)-0.01*i)

        # plot h
        train_m = model(test_x)
        ax.plot(test_x.numpy(), train_m.numpy(), 'g')

        # plot 95% credibility region
        upper = torch.squeeze(test_f, 1) + 2*cov_f.pow(0.5)
        lower = torch.squeeze(test_f, 1) - 2*cov_f.pow(0.5)
        ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        # plot predictions
        ax.plot(test_x.numpy(), test_f.detach().numpy(), 'b')
        ax.set_ylim([-2, 2])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        plt.show()
