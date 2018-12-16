import torch
import math
from matplotlib import pyplot as plt
import gp_regression_cpy as gpr
import numpy as np

def truefunc(omega,points):
    return torch.sin(torch.squeeze(points,1) * omega)

def trufunc_int(omega,points_lims):
    return -torch.cos(omega*points_lims[:,1])/omega+torch.cos(omega*points_lims[:,0])/omega

noise_std=0.01
n = 10
omega=2

# inputs (integral limits)
train_x = torch.rand(n,2)*2*math.pi

# output (integral measurements)
train_y = trufunc_int(omega,train_x) + torch.randn(n) * noise_std

# test points
test_x = torch.Tensor(100, 1)
test_x[:, 0] = torch.linspace(0, 2*math.pi, 100)

model = gpr.GP_1D_int(sigma_f=1.0, lengthscale=1, sigma_n=1)
# c, v = model(train_x, train_y)

print(model)

nLL = gpr.NegMarginalLogLikelihood()  # this is the loss function

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.005)

training_iterations = 200


def train():
    for i in range(training_iterations):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        c, v = model(train_x, train_y)
        # Calc loss and backprop derivatives
        loss = nLL(train_y, c, v)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()


train()
print(model)

# now make predictions
test_f, cov_f = model(train_x,train_y,test_x)

with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 3))
    # Plot true function as solid black
    ax.plot(test_x.numpy(), truefunc(omega,test_x).numpy(), 'k')
    # plot integral regions
    for i in range(n):
        ax.plot(train_x[i,:].numpy(),np.zeros(2)+0.02*i)
    # Plot training data as black stars
    #ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    upper = torch.squeeze(test_f, 1) + cov_f.pow(0.5)
    lower = torch.squeeze(test_f, 1) - cov_f.pow(0.5)
    # plot predictions
    ax.plot(test_x.numpy(), test_f.numpy(), 'b')
    ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-2, 2])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()

