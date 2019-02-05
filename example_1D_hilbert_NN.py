import torch
import math
from matplotlib import pyplot as plt
import gp_regression_hilbert as gprh

# added these lines to include PyTorch-LBFGS
import sys
sys.path.append('./functions/')
from LBFGS import FullBatchLBFGS

def truefunc(omega,points):
    # return torch.cos(torch.squeeze(points, 1) * omega)
    out=points.clone()
    out[points<0.5]=-1
    out[points>0.5]=1
    # out[points>0.7]=-1
    return out.view(-1)

omega=8*math.pi
noise_std=0.05
n = 50
train_x = torch.linspace(0,1, n).view(n,1)
train_y = truefunc(omega,train_x) + torch.randn(n) * noise_std

nt = 1000
test_x = torch.linspace(0, 1, nt).view(nt, 1)

# set appr params
m = 60 # nr of basis functions

class DeepGP(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(DeepGP, self).__init__()
        self.linear1 = torch.nn.Linear(1, 30)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(30, 30)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(30, 6)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(6, 1)
        self.tanh4 = torch.nn.Sigmoid()
        self.gp = gprh.GP_1D(sigma_f=1.0, lengthscale=0.05, sigma_n=2*noise_std)

    def forward(self, x_train, y_train=None, m=None, x_test=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = x_train.clone()
        h = self.linear1(h)
        h = self.tanh1(h)
        h = self.linear2(h)
        h = self.tanh2(h)
        h = self.linear3(h)
        h = self.tanh3(h)
        h = self.linear4(h)
        h = self.tanh4(h)
        if x_test is not None:
            h2 = x_test.clone()
            h2 = self.linear1(h2)
            h2 = self.tanh1(h2)
            h2 = self.linear2(h2)
            h2 = self.tanh2(h2)
            h2 = self.linear3(h2)
            h2 = self.tanh3(h2)
            h2 = self.linear4(h2)
            h2 = self.tanh4(h2)
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(h,y_train,m,h2)
        else:
            out = h
        return out


model = DeepGP()

model.gp.covtype="se"
nLL = gprh.NegMarginalLogLikelihood_deep(model.gp.covtype,model.gp.nu)  # this is the loss function

# PICK BETTER OPTIMISER
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)

training_iterations = 700

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.99,min_lr=1e-6)

#def train():
for i in range(training_iterations):
    # Zero backprop gradients
    optimizer.zero_grad()

    # get train_m
    train_m = model(train_x)

    loss = nLL(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, train_m, train_y, m)
    loss.backward()

    optimizer.step()

    scheduler.step(i)

    print('Iter %d/%d - Loss: %.3f - sigf: %.3f - l: %.3f - sign: %.3f' % (i + 1, training_iterations, loss.item(), model.gp.log_sigma_f.exp(), model.gp.log_lengthscale.exp(), model.gp.log_sigma_n.exp() ))



#train()
print(model)

# now make predictions
test_f, cov_f = model(train_x, train_y, m, test_x)

with torch.no_grad():
    fplot, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'r*')

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
    #ax.legend(['Observed Data', 'True', 'Predicted'])
    plt.show()
