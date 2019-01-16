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

omega=8*math.pi
noise_std=0.01
n = 300
train_x = torch.Tensor(n, 1)
train_x[:, 0] = torch.linspace(0,1, n)
train_y = truefunc(omega,train_x) + torch.randn(n) * noise_std

test_x = torch.Tensor(100, 1)
test_x[:, 0] = torch.linspace(0, 1, 100)

# set appr params
m = 60 # nr of basis functions

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
        self.gp = gprh.GP_1D(sigma_f=1.0, lengthscale=0.05, sigma_n=noise_std)

    def forward(self, x_train, y_train=None, m=None, x_test=None):
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
        if y_train is not None:
            out = self.gp(h,y_train,m,h2)
        else:
            out = h
        return out


model = DeepGP()

nLL = gprh.NegMarginalLogLikelihood_deep()  # this is the loss function

# PICK BETTER OPTIMISER
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)

training_iterations = 500


def train():
    for i in range(training_iterations):
        # Zero backprop gradients
        optimizer.zero_grad()

        # get train_m
        train_m = model(train_x)

        # Calc loss and backprop derivatives
        loss = nLL(model.gp.sigma_f, model.gp.lengthscale, model.gp.sigma_n, train_m, train_y, m)
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
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

    # Plot true function as solid black
    ax.plot(test_x.numpy(), truefunc(omega,test_x).numpy(), 'k')

    # plot h
    train_m = model(test_x)
    ax.plot(test_x.numpy(), train_m.numpy(), 'g')

    # plot 95% credibility region
    upper = torch.squeeze(test_f, 1) + 2*cov_f.pow(0.5)
    lower = torch.squeeze(test_f, 1) - 2*cov_f.pow(0.5)
    ax.fill_between(torch.squeeze(test_x,1).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

    # plot predictions
    ax.plot(test_x.numpy(), test_f.numpy(), 'b')

    #ax.set_ylim([-2, 2])
    ax.legend(['Observed Data', 'True', 'Predicted'])
    plt.show()
