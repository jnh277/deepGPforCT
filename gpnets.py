import torch
import math
import gp_regression_hilbert as gprh

##################################### 1D input #########################################################################
### nets of type 1-1-gp
class gpnet1_1_1(torch.nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description:
        """
        super(gpnet1_1_1, self).__init__()
        self.linear1 = torch.nn.Linear(1, 30)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(30, 30)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(30, 6)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(6, 1)
        self.sigm = torch.nn.Sigmoid()
        self.scale = torch.nn.Parameter(torch.Tensor([1.0]))

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h11 = x_train.clone()
            h11 = self.linear1(h11)
            h11 = self.tanh1(h11)
            h11 = self.linear2(h11)
            h11 = self.tanh2(h11)
            h11 = self.linear3(h11)
            h11 = self.tanh3(h11)
            h11 = self.linear4(h11)
            h11 = self.sigm(h11)
            h11 = self.scale * h11

            h = h11

        if x_test is not None:
            h21 = x_test.clone()
            h21 = self.linear1(h21)
            h21 = self.tanh1(h21)
            h21 = self.linear2(h21)
            h21 = self.tanh2(h21)
            h21 = self.linear3(h21)
            h21 = self.tanh3(h21)
            h21 = self.linear4(h21)
            h21 = self.sigm(h21)
            h21 = self.scale * h21

            h2 = h21
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out

### nets of type 1-2-gp
class gpnet1_2_1(torch.nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description:
        """
        super(gpnet1_2_1, self).__init__()
        self.linear1 = torch.nn.Linear(1, 30)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(30, 30)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(30, 6)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(6, 1)
        self.sigm = torch.nn.Sigmoid()
        self.scale = torch.nn.Parameter(torch.Tensor([1.0]))

        self.linear21 = torch.nn.Linear(1, 30)
        self.tanh21 = torch.nn.Tanh()
        self.linear22 = torch.nn.Linear(30, 6)
        self.tanh22 = torch.nn.Tanh()
        self.linear23 = torch.nn.Linear(6, 1)

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h11 = x_train.clone()
            h11 = self.linear1(h11)
            h11 = self.tanh1(h11)
            h11 = self.linear2(h11)
            h11 = self.tanh2(h11)
            h11 = self.linear3(h11)
            h11 = self.tanh3(h11)
            h11 = self.linear4(h11)
            h11 = self.sigm(h11)
            h11 = self.scale * h11

            h12 = x_train.clone()
            h12 = self.linear21(h12)
            h12 = self.tanh21(h12)
            h12 = self.linear22(h12)
            h12 = self.tanh22(h12)
            h12 = self.linear23(h12)

            h = torch.cat((h11,h12),1)

        if x_test is not None:
            h21 = x_test.clone()
            h21 = self.linear1(h21)
            h21 = self.tanh1(h21)
            h21 = self.linear2(h21)
            h21 = self.tanh2(h21)
            h21 = self.linear3(h21)
            h21 = self.tanh3(h21)
            h21 = self.linear4(h21)
            h21 = self.sigm(h21)
            h21 = self.scale * h21

            h22 = x_test.clone()
            h22 = self.linear21(h22)
            h22 = self.tanh21(h22)
            h22 = self.linear22(h22)
            h22 = self.tanh22(h22)
            h22 = self.linear23(h22)

            h2 = torch.cat((h21,h22),1)

        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


##################################### 2D input #########################################################################
