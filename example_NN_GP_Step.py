import math
import torch
import gp_regression as gpr
import torch.nn as nn
import gpytorch
from matplotlib import pyplot as plt

from math import floor

n = 40
train_x = torch.Tensor(n, 1)
train_x[:, 0] = torch.linspace(0, 1, n)
train_y = 0.5*torch.sin(torch.squeeze(train_x, 1) * (3 * math.pi))
train_y[torch.squeeze(train_x, 1) > 0.5] = train_y[torch.squeeze(train_x, 1) > 0.5] + 1
train_y[torch.squeeze(train_x, 1) <= 0.5] = train_y[torch.squeeze(train_x, 1) <= 0.5] - 1
train_y = train_y + torch.randn(train_y.size()) * 0.2

test_x = torch.Tensor(100, 1)
test_x[:, 0] = torch.linspace(0, 1, 100)


data_dim = train_x.size(-1)





class DeepGP(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(DeepGP, self).__init__()
        self.linear1 = torch.nn.Linear(1, 100)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(100, 6)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(6, 1)
        self.tanh3 = torch.nn.Sigmoid()
        self.gp = gpr.GP_SE(sigma_f=1.0, lengthscale=[1, 1], sigma_n=1)

    def forward(self, x_train, y_train, x_test=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = self.linear1(x_train)
        h = self.tanh1(h)
        h = self.linear2(h)
        h = self.tanh2(h)
        h = self.linear3(h)
        h = self.tanh3(h)
        if x_test is not None:
            h2 = self.linear1(x_test)
            h2 = self.tanh1(h2)
            h2 = self.linear2(h2)
            h2 = self.tanh2(h2)
            h2 = self.linear3(h2)
            h2 = self.tanh3(h2)
        else:
            h2 = None
        out = self.gp(h,y_train,h2)
        return out


deepGP = DeepGP()

nLL = gpr.NegMarginalLogLikelihood()  # this is the loss function

optimizer = torch.optim.Adam([
    {'params': deepGP.parameters()},
], lr=0.005)

training_iterations = 300


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
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    upper = torch.squeeze(test_f, 1) + cov_f.pow(0.5)
    lower = torch.squeeze(test_f, 1) - cov_f.pow(0.5)
    # plot predictions
    ax.plot(test_x.numpy(), test_f.numpy(), 'b')
    ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-2, 2])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()
