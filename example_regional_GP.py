import math
import torch
import gp_regression as gpr
from matplotlib import pyplot as plt


n = 50
train_x = torch.Tensor(n, 1)
train_x[:, 0] = torch.linspace(0, 1, n)
train_y = 0.25*torch.sin(torch.squeeze(train_x, 1) * (3 * math.pi))
train_y[torch.squeeze(train_x, 1) > 0.5] = train_y[torch.squeeze(train_x, 1) > 0.5] + 1
# train_y[torch.squeeze(train_x, 1) <= 0.5] = train_y[torch.squeeze(train_x, 1) <= 0.5]
train_y = train_y + torch.randn(train_y.size()) * 0.05

train_body = train_x > 0.5

test_x = torch.Tensor(100, 1)
test_x[:, 0] = torch.linspace(0, 1, 100)
test_body = test_x > 0.5


data_dim = train_x.size(-1)
lengthscales = torch.ones(1)
model = gpr.GP_SE_R(sigma_f=1.0, lengthscale=lengthscales, sigma_n=0.5)

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
        c, v = model(train_x, train_y, train_body)
        # Calc loss and backprop derivatives
        loss = nLL(train_y, c, v)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()


train()


test_f, cov_f = model(train_x, train_y, train_body, test_x, test_body)

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