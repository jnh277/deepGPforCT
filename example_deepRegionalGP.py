import math
import torch
import gp_regression as gpr
from matplotlib import pyplot as plt


n = 20
train_x = torch.Tensor(n, 1)
train_x[:, 0] = torch.linspace(0, 1, n)
train_y = 0.5*torch.sin(torch.squeeze(train_x, 1) * (3 * math.pi))
train_y[torch.squeeze(train_x, 1) > 0.5] = train_y[torch.squeeze(train_x, 1) > 0.5] + 1
train_y[torch.squeeze(train_x, 1) <= 0.5] = train_y[torch.squeeze(train_x, 1) <= 0.5]-1
train_y = train_y + torch.randn(train_y.size()) * 0.2

train_body = train_x > 0.5

test_x = torch.Tensor(100, 1)
test_x[:, 0] = torch.linspace(0, 1.0, 100)
test_body = test_x > 0.5


data_dim = train_x.size(-1)
lengthscales = torch.ones(1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 10))
        self.add_module('Tanh1', torch.nn.Tanh())
        self.add_module('linear2', torch.nn.Linear(10, 3))
        self.add_module('Tanh2', torch.nn.Tanh())
        self.add_module('linear3', torch.nn.Linear(3, 1))
        self.add_module('Sigmoid3', torch.nn.Sigmoid())     # final layer should output between 0 and 1



feature_extractor = LargeFeatureExtractor()


class DeepGP(torch.nn.Module):
    def __init__(self,data_dim,sigma_f=None,lengthscale=None,sigma_n=None):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(DeepGP, self).__init__()
        if sigma_f is None:
            sigma_f = torch.ones(1)
        if lengthscale is None:
            lengthscale = torch.ones(data_dim)
        if sigma_n is None:
            sigma_n = torch.ones(1)
        self.feature_extractor = feature_extractor
        self.gp = gpr.GP_SE_R(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

    def forward(self, x_train, y_train, x_test=None,classify=False):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = self.feature_extractor(x_train)
        if x_test is not None:
            h2 = self.feature_extractor(x_test)
        else:
            h2 = None
        out = self.gp(x_train,y_train,h,x_test,h2,classify=classify)
        return out



model = DeepGP(data_dim=data_dim,sigma_f=1.0, lengthscale=lengthscales, sigma_n=0.2)

nLL = gpr.NegMarginalLogLikelihood()  # this is the loss function


optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.gp.parameters()}
], lr=0.005)

training_iterations = 800


def train(classify=False):
    for i in range(training_iterations):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        c, v = model(train_x, train_y,classify=classify)
        # Calc loss and backprop derivatives
        loss = nLL(train_y, c, v)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()


train()

# now set to classify = true and train only the gp parameters
optimizer = torch.optim.Adam([
    {'params': model.gp.parameters()}
], lr=0.005)

training_iterations = 200
train(classify=True)

test_f, cov_f = model(train_x, train_y, test_x, classify=True)
est_B = feature_extractor(test_x)
classified = est_B > 0.5

with torch.no_grad():
    fplot, ax = plt.subplots(2, 1, figsize=(4, 4))
    # Plot training data as black stars
    ax[0].plot(train_x.numpy(), train_y.numpy(), 'k*')
    upper = torch.squeeze(test_f, 1) + cov_f.pow(0.5)*2
    lower = torch.squeeze(test_f, 1) - cov_f.pow(0.5)*2
    # plot predictions
    ax[0].plot(test_x.numpy(), test_f.numpy(), 'b')
    ax[0].fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax[0].set_ylim([-2, 2])
    ax[0].legend(['Observed Data', 'Mean', 'Confidence'])

    # fplot, ax = plt.subplots(1, 1, figsize=(4, 3))
    # Plot training data as black stars
    ax[1].plot(test_x.numpy(), test_body.numpy(), 'b')
    ax[1].plot(test_x.numpy(), est_B.detach().numpy(), 'g')
    ax[1].plot(test_x.numpy(), classified.detach().numpy(), '--r')
    # ax[1].title(['Regions'])
    ax[1].legend(['true regions', 'estimated','classified as'])
    plt.show()