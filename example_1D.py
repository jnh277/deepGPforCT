import torch
import math
from matplotlib import pyplot as plt
import gp_regression as gpr


n = 10
train_x = torch.Tensor(n, 1)
train_x[:, 0] = torch.linspace(0, 2 * math.pi, n)
train_y = 1*torch.sin(torch.squeeze(train_x, 1)) + torch.randn(n) * 0.2

test_x = torch.Tensor(100, 1)
test_x[:, 0] = torch.linspace(0, 2*math.pi, 100)

model = gpr.GP_1D(sigma_f=1.0, lengthscale=0.5, sigma_n=0.2)
test_f, cov_f = model(train_x, train_y, test_x)

print(model)

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