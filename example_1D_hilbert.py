import torch
import math
from matplotlib import pyplot as plt
import gp_regression_hilbert as gprh

def truefunc(omega,points):
    # return torch.cos(torch.squeeze(points, 1) * omega)
    out=points.clone()
    out[points<0.5]=-1
    out[points>0.5]=1
    return out.view(-1)

omega=4*math.pi
noise_std=0.01
n = 200
train_x = torch.Tensor(n, 1)
train_x[:, 0] = torch.linspace(0,1, n)
train_y = truefunc(omega,train_x) + torch.randn(n) * noise_std

nt=1000
test_x = torch.linspace(0, 1, nt).view(nt,1)

# set appr params
m = 60 # nr of basis functions

model = gprh.GP_1D(sigma_f=1, lengthscale=0.05, sigma_n=noise_std, covtype="matern")

# print(model)

nLL = gprh.NegMarginalLogLikelihood(model.covtype,model.nu)  # this is the loss function

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.05)

training_iterations = 600


def train():
    for i in range(training_iterations):
        # Zero backprop gradients
        optimizer.zero_grad()

        # Calc loss and backprop derivatives
        loss = nLL(model.log_sigma_f, model.log_lengthscale, model.log_sigma_n, train_x, train_y, m)
        loss.backward()

        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

train()
print(model)

# now make predictions
test_f, cov_f = model(train_x, train_y, m, test_x)

with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'r*')

    # Plot true function as solid black
    ax.plot(test_x.numpy(), truefunc(omega,test_x).numpy(), 'k')

    # plot 95% credibility region
    upper = torch.squeeze(test_f, 1) + 2*cov_f.pow(0.5)
    lower = torch.squeeze(test_f, 1) - 2*cov_f.pow(0.5)
    ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

    # plot predictions
    ax.plot(test_x.numpy(), test_f.numpy(), 'b')

    #ax.set_ylim([-2, 2])
    #ax.legend(['Observed Data', 'True', 'Predicted'])
    plt.show()
