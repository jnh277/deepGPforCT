import torch
import math
from matplotlib import pyplot as plt
import gp_regression_hilbert as gprh
import integration as int
import numpy as np

def cosfunc(omega,points):
    return torch.cos(torch.squeeze(points, 1) * omega)

def cosfunc_int(omega,points_lims):
    return torch.sin(omega*points_lims[:,1])/omega - torch.sin(omega*points_lims[:,0])/omega

def stepfunc(points):
    # return torch.cos(torch.squeeze(points, 1) * omega)
    out=points.clone()
    out[points<0.5]=-1
    out[points>0.5]=1
    return out.view(-1)

integral=True
points=False

if integral:
    # INTEGRAL INPUTS
    omega=4*math.pi
    noise_std=0.01
    n = 20 # 300

    # inputs (integral limits)
    train_x = torch.rand(n,2)

    # output (integral measurements)
    train_y = cosfunc_int(omega,train_x) + torch.randn(n) * noise_std

    # test points
    test_x = torch.Tensor(100, 1)
    test_x[:, 0] = torch.linspace(0, 1, 100)
    # END INTEGRAL INPUTS

if points:
    # POINT INPUTS
    omega=4*math.pi
    noise_std=0.01
    n = 50
    train_x = torch.Tensor(n, 1)
    train_x[:, 0] = torch.linspace(0,1, n)
    train_y = cosfunc(omega,train_x) + torch.randn(n) * noise_std

    test_x = torch.Tensor(100, 1)
    test_x[:, 0] = torch.linspace(0, 1, 100)
    # END OF POINT INPUTS

# set appr params
m = 50 # 60 # nr of basis functions
L = 3  # hard-coded
# END POINT INPUT

class DeepGP(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(DeepGP, self).__init__()
        # self.linear1 = torch.nn.Linear(1, 10)
        # self.tanh1 = torch.nn.Tanh()
        # self.linear2 = torch.nn.Linear(10, 1)
        # self.tanh2 = torch.nn.Tanh()
        # self.linear3 = torch.nn.Linear(1, 1)
        # self.tanh3 = torch.nn.Sigmoid()
        self.gp = gprh.GP_1D_new(sigma_f=1.0, lengthscale=0.05, sigma_n=noise_std)

    def forward(self, x_train=None, y_train=None, phi=None, m=None, L=None, x_test=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if x_train is not None:
            h=x_train
            # h = self.linear1(x_train)
            # h = self.tanh1(h)
            # h = self.linear2(h)
            # h = self.tanh2(h)
            # h = self.linear3(h)
            # h = self.tanh3(h)
        if x_test is not None:
            h=x_test
            # h2 = self.linear1(x_test)
            # h2 = self.tanh1(h2)
            # h2 = self.linear2(h2)
            # h2 = self.tanh2(h2)
            # h2 = self.linear3(h2)
            # h2 = self.tanh3(h2)
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,m,L,h2)
        else:
            out = h
        return out


model = DeepGP()  # define the model

nLL = gprh.NegMarginalLogLikelihood_deep_intMeas()  # this is the loss function

# PICK BETTER OPTIMISER
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
],lr=0.01)

# set numerical integration tool
simpsons = int.Simpsons(fcount_out=False, fcount_max=1e3, hmin=None)

# create an index vector, index=[1 2 3...]
index = torch.empty(1, m)
for i in range(m):
    index[0, i]=i+1

training_iterations = 3000
def train():
    for i in range(training_iterations):
        def closure():
            # Zero backprop gradients
            optimizer.zero_grad()

            if points:
                train_m = model(train_x)
                phi = ( 1/math.sqrt(L) ) * torch.sin(math.pi*index*(train_m+L)*0.5/L) # basis functions

            if integral:
                # calculate phi using numerical integration
                phi = torch.zeros(n,m)
                for k in range(n):
                    for q in range(m):
                        phi[k,q] = simpsons(lambda x: ( 1/math.sqrt(L) )*torch.sin(math.pi*(q+1)*(model(x.view(1))+L)*0.5/L), train_x[k,0].view(1), train_x[k,1].view(1), 1e-6)

                # 1 for-loop (not faster)
                # phi = torch.zeros(n*m,1)
                # for k in range(n*m):
                #     j=math.floor(k/m)
                #     q=k%m
                #     func = lambda x: ( 1/math.sqrt(L) )*torch.sin(math.pi*(q+1)*(model(x.view(1))+L)*0.5/L)
                #     phi[k,0] = simpsons(func, m_l[j], m_u[j], 1e-6)

            # Calc loss and backprop derivatives
            loss = nLL(model.gp.sigma_f, model.gp.lengthscale, model.gp.sigma_n, phi.view(m*n,1), train_y, m, L)
            loss.backward()
            return loss

        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, closure().item()))
        optimizer.step(closure)

train()
print(model)

if integral:
    # calculate final phi
    phi = torch.zeros(n,m)
    for k in range(n):
        for q in range(m):
            phi[k,q] = simpsons(lambda x: ( 1/math.sqrt(L) )*torch.sin(math.pi*(q+1)*(model(x.view(1))+L)*0.5/L), train_x[k,0].view(1), train_x[k,1].view(1), 1e-6)

if points:
    train_m = model(train_x)
    phi = ( 1/math.sqrt(L) ) * torch.sin(math.pi*index*(train_m+L)*0.5/L) # basis functions

# now make predictions
test_f, cov_f = model(None, train_y, phi, m, L, test_x)

if points:
    with torch.no_grad():
        fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

        # Plot true function as solid black
        ax.plot(test_x.numpy(), cosfunc(omega,test_x).numpy(), 'k')

        # # plot h
        # train_m = model(test_x)
        # ax.plot(test_x.numpy(), train_m.numpy(), 'g')

        # plot 95% credibility region
        upper = torch.squeeze(test_f, 1) + 2*cov_f.pow(0.5)
        lower = torch.squeeze(test_f, 1) - 2*cov_f.pow(0.5)
        ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        # plot predictions
        ax.plot(test_x.numpy(), test_f.numpy(), 'b')

        #ax.set_ylim([-2, 2])
        ax.legend(['Observed Data', 'True', 'Predicted'])
        plt.show()

if integral:
    with torch.no_grad():
        fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot true function as solid black
        ax.plot(test_x.numpy(), cosfunc(omega,test_x).numpy(), 'k')

        # plot integral regions
        for i in range(n):
            ax.plot(train_x[i,:].numpy(),np.zeros(2)+0.02*i)

        # plot h
        train_m = model(test_x)
        ax.plot(test_x.numpy(), train_m.numpy(), 'g')

        # plot 95% credibility region
        upper = torch.squeeze(test_f, 1) + 2*cov_f.pow(0.5)
        lower = torch.squeeze(test_f, 1) - 2*cov_f.pow(0.5)
        ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        # plot predictions
        ax.plot(test_x.numpy(), test_f.numpy(), 'b')
        ax.set_ylim([-2, 2])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

        plt.show()
