import math
import torch
import gp_regression as gpr
import torch.nn as nn
import gpytorch
from matplotlib import pyplot as plt

from math import floor

def truefunc(omega,points):
    step=0.5
    g = torch.squeeze(points,1).clone()
    y = g.clone()
    p1 = g >= step; y1 = y[p1]
    p2 = g < step; y2 = y[p2]
    y[p1] = torch.sin(y1* 0.5 * omega)- 1.0
    y[p2] = torch.sin(y2* 5 * omega)*torch.exp(torch.sin(-10*y2)) + 1.0
    y[g<0.15] -= 2.0
    return y

# def truefunc(omega,points):
#     step=0.5
#     g = torch.squeeze(points,1).clone()
#     y = g.clone()
#     p1 = g >= step; y1 = y[p1]
#     p2 = g < step; y2 = y[p2]
#     y[p1] = torch.sin(y1 * omega)- 1.0
#     y[p2] = torch.sin(y2* omega) + 1.0
#     # y[g<0.15] -= 2.0
#     return y

omega=4*math.pi
noise_std=0.05
n = 200
train_x = torch.linspace(0,1, n).view(n,1)
train_y = truefunc(omega,train_x) + torch.randn(n) * noise_std

nt = 1000
test_x = torch.linspace(0, 1, nt).view(nt, 1)


data_dim = train_x.size(-1)

class DeepGP(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(DeepGP, self).__init__()

        self.linear1 = torch.nn.Linear(1, 30)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(30, 6)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(6, 1)
        self.tanh3 = torch.nn.Sigmoid()
        self.scale = torch.nn.Parameter(torch.Tensor([1.0]))

        self.linear21 = torch.nn.Linear(1, 30)
        self.tanh21 = torch.nn.Tanh()
        self.linear22 = torch.nn.Linear(30, 6)
        self.tanh22 = torch.nn.Tanh()
        self.linear23 = torch.nn.Linear(6, 1)

        self.gp = gpr.GP_SE(sigma_f=1.0, lengthscale=[1, 1], sigma_n=1)

    def forward(self, x_train, y_train=None, x_test=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h11 = x_train.clone()
        h11 = self.linear1(h11)
        h11 = self.tanh1(h11)
        h11 = self.linear2(h11)
        h11 = self.tanh2(h11)
        h11 = self.linear3(h11)
        h11 = self.tanh3(h11)
        h11 = self.scale * h11

        h12 = x_train.clone()
        h12 = self.linear21(h12)
        h12 = self.tanh21(h12)
        h12 = self.linear22(h12)
        h12 = self.tanh22(h12)
        h12 = self.linear23(h12)

        h = torch.cat((h11,h12),1)
        if x_test is not None:
            h21 = x_test.clone()
            h21 = self.linear1(h21)
            h21 = self.tanh1(h21)
            h21 = self.linear2(h21)
            h21 = self.tanh2(h21)
            h21 = self.linear3(h21)
            h21 = self.tanh3(h21)
            h21 = self.scale * h21

            h22 = x_test.clone()
            h22 = self.linear21(h22)
            h22 = self.tanh21(h22)
            h22 = self.linear22(h22)
            h22 = self.tanh22(h22)
            h22 = self.linear23(h22)

            h2 = torch.cat((h21,h22),1)
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(h,y_train,h2)
        else:
            out = h
        return out


deepGP = DeepGP()

nLL = gpr.NegMarginalLogLikelihood()  # this is the loss function

optimizer = torch.optim.Adam([
    {'params': deepGP.parameters()},
], lr=0.005)

training_iterations = 2500


def train():
    for i in range(training_iterations):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        c, v = deepGP(train_x, train_y)
        # Calc loss and backprop derivatives
        loss = nLL(train_y, c, v)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()


train()

# now make predictions
test_f, cov_f = deepGP(train_x, train_y, test_x)

with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Plot training data as red stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'r*')
    upper = torch.squeeze(test_f, 1) + cov_f.pow(0.5)
    lower = torch.squeeze(test_f, 1) - cov_f.pow(0.5)

    # Plot true function as solid black
    ax.plot(test_x.numpy(), truefunc(omega,test_x).numpy(), 'k')

    # # plot latent outputs
    # train_m = deepGP(test_x)
    # ax.plot(test_x.numpy(), train_m[:,0].numpy(), 'g')
    # ax.plot(test_x.numpy(), train_m[:,1].numpy(), 'g')

    # plot predictions
    ax.plot(test_x.numpy(), test_f.numpy(), 'b')
    ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-2, 2])
    ax.legend(['Observed Data', 'True function', 'Mean', 'Confidence'])
    plt.show()
