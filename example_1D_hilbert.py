import torch
import math
from matplotlib import pyplot as plt
import gp_regression_hilbert as gprh

def truefunc(omega,points):
    return torch.cos(torch.squeeze(points, 1) * omega)

omega=8*math.pi
noise_std=0.2
n = 300
train_x = torch.Tensor(n, 1)
train_x[:, 0] = torch.linspace(0,1, n)
train_y = truefunc(omega,train_x) + torch.randn(n) * noise_std

test_x = torch.Tensor(100, 1)
test_x[:, 0] = torch.linspace(0, 1, 100)

# set appr params
m = 40 # nr of basis functions
L = 2 # domain expansion

model = gprh.GP_1D(sigma_f=2.0, lengthscale=0.1, sigma_n=2*noise_std)
#r, v, inv_lambda_diag, phi, bigphi = model(train_x, train_y, m, L)

# print(model)

nLL = gprh.NegMarginalLogLikelihood()  # this is the loss function

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)

training_iterations = 400


def train():
    for i in range(training_iterations):
        # Zero backprop gradients
        optimizer.zero_grad()

        # Calc loss and backprop derivatives
        loss = nLL(model.sigma_f, model.lengthscale, model.sigma_n, train_x, train_y, m, L)
        loss.backward()

        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

train()
print(model)

# now make predictions
test_f, cov_f = model(train_x, train_y, m, L, test_x)
#
with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

    # Plot true function as solid black
    ax.plot(test_x.numpy(), truefunc(omega,test_x).numpy(), 'k')

    # plot 95% credibility region
    upper = torch.squeeze(test_f, 1) + 2*cov_f.pow(0.5)
    lower = torch.squeeze(test_f, 1) - 2*cov_f.pow(0.5)
    ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

    # plot predictions
    ax.plot(test_x.numpy(), test_f.numpy(), 'b')

    #ax.set_ylim([-2, 2])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()
