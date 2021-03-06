import torch
import math
from matplotlib import pyplot as plt
import gp_regression as gpr


def truefunc(omega,points):
    # return torch.sin(torch.squeeze(points, 1) * omega)
    out=points.clone()
    out[points<0.5]=-1
    out[points>0.5]=1
    return out.view(-1)

omega=4*math.pi
noise_std=1e-2
n = 100
train_x = torch.linspace(0,1, n).view(n,1)
train_y = truefunc(omega,train_x) + torch.randn(n) * noise_std

nt = 1000
test_x = torch.linspace(0, 1, nt).view(nt,1)

class senn(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(senn, self).__init__()
        self.gp_nn = gpr.GP_1D_NN(sigmaentr=[10,50], sigma_n=noise_std)
        self.gp_se = gpr.GP_1D(sigma_f=1.0, lengthscale=1.0, sigma_n=noise_std)

    def forward(self, x_train, y_train, x_test=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = self.gp_nn(x_train,y_train,x_train)
        if x_test is not None:
            h2 = self.gp_nn(x_train,y_train,x_test)
            # out=h2
        else:
            h2 = None
            # out=h
        out = self.gp_se(h,y_train,h2)
        return out

model=senn()
print(model)

nLL = gpr.NegMarginalLogLikelihood()  # this is the loss function

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.001)

training_iterations = 1000


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

# now make predictions
test_f,cov_f= model(train_x,train_y,test_x)

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
