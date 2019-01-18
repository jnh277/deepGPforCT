import torch
import math
from matplotlib import pyplot as plt
import gp_regression as gpr
import numpy as np


def truefunc(omega,points,step):
    y = torch.sin(torch.squeeze(points,1) * omega)
    y[torch.squeeze(points, 1) >= step] += -1.0
    y[torch.squeeze(points, 1) < step] += +1.0
    return y

def trufunc_int(omega,points_lims,step):
    a0 = torch.clamp(points_lims[:, 0], max=step)
    a1 = torch.clamp(points_lims[:, 1], max=step)
    a2 = torch.clamp(points_lims[:, 0], min=step)
    a3 = torch.clamp(points_lims[:, 1], min=step)
    # if omega < 1e-8:
    p1 = 1.0*(a1-a0)
    p2 = - 1.0 * (a3 - a2)
    # else:
    #     p1 = -torch.cos(omega*a1)/omega+torch.cos(omega*a0)/omega+1.0*(a1-a0)
    #     p2 = -torch.cos(omega * a3) / omega + torch.cos(omega * a2) / omega - 1.0 * (a3 - a2)
    return p1+p2


noise_std = 0.01
n = 50
omega = 0.0
step = 0.5

# inputs (integral limits)
train_x = torch.empty(n, 2)
train_x[:, 0] = -0.125 + torch.squeeze(torch.rand(n,1))
train_x[:, 1] = train_x[:,0]+0.25*torch.squeeze(torch.rand(n,1))

# output (integral measurements)
train_y = trufunc_int(omega,train_x,step) + torch.randn(n) * noise_std

# test points
test_x = torch.Tensor(100, 1)
test_x[:, 0] = torch.linspace(0, 1, 100)

model = gpr.DeepGP(sigma_f=1, lengthscale=1.0, sigma_n=0.1)

print(model)

nLL = gpr.NegMarginalLogLikelihood()  # this is the loss function

optimizer = torch.optim.Adam([
    # {'params': model.reg_nn.parameters()},
    {'params': model.parameters()}
], lr=0.01)

training_iterations = 3000

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, verbose=True, factor=0.5,min_lr=1e-6)
def train():
    count = 0
    for i in range(training_iterations):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        c, v = model(train_x, train_y)
        # Calc loss and backprop derivatives
        loss = nLL(train_y, c, v)
        loss.backward()
        if i > 0:
            if loss > loss_old-1e-2:
                count += 1
                print('count = {}'.format(count))
                if count > 110:
                    break
            else:
                count = 0
        loss_old=loss
        # loss.backward()
        optimizer.step()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        # scheduler.step(i)







train()
print(model)

# now make predictions
test_f, cov_f = model(train_x, train_y, test_x)

with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 3))
    # Plot true function as solid black
    ax.plot(test_x.numpy(), truefunc(omega,test_x,step).numpy(), 'k')
    # plot integral regions
    for i in range(n):
        ax.plot(train_x[i,:].numpy(),(train_y[i] * torch.ones(2)).numpy())
    # Plot training data as black stars
    #ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    upper = torch.squeeze(test_f, 1) + cov_f.pow(0.5)
    lower = torch.squeeze(test_f, 1) - cov_f.pow(0.5)
    # plot predictions
    ax.plot(test_x.numpy(), test_f.numpy(), 'b')
    ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-2, 2])
    ax.legend(['True', 'Mean', 'Confidence'])
    h = model.feature_extractor(test_x)
    ax.plot(test_x.numpy(), h.detach().numpy(), 'g')
    plt.show()



