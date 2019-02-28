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
### nets of type 2-1-gp
class gpnet2_1_1(torch.nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description:
        """
        super(gpnet2_1_1, self).__init__()
        self.linear1 = torch.nn.Linear(2, 30)
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


class gpnet2_1_2(torch.nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description: smaller than above
        """
        super(gpnet2_1_2, self).__init__()
        self.linear1 = torch.nn.Linear(2, 20)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(20, 6)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(6, 1)
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


class gpnet2_1_3(torch.nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description:
        """
        super(gpnet2_1_3, self).__init__()
        self.linear1 = torch.nn.Linear(2, 80)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(80, 80)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(80, 80)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(80, 80)
        self.tanh4 = torch.nn.Tanh()
        self.linear5 = torch.nn.Linear(80, 1)
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
            h11 = self.tanh4(h11)
            h11 = self.linear5(h11)
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
            h21 = self.tanh4(h21)
            h21 = self.linear5(h21)
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


class gpnet2_1_4(torch.nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description: deeper than the previous net
        """
        super(gpnet2_1_4, self).__init__()
        self.linear1 = torch.nn.Linear(2, 50)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(50, 50)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(50, 50)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(50, 50)
        self.tanh4 = torch.nn.Tanh()
        self.linear5 = torch.nn.Linear(50, 50)
        self.tanh5 = torch.nn.Tanh()
        self.linear6 = torch.nn.Linear(50, 50)
        self.tanh6 = torch.nn.Tanh()
        self.linear7 = torch.nn.Linear(50, 40)
        self.tanh7 = torch.nn.Tanh()
        self.linear8 = torch.nn.Linear(40, 1)
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
            h11 = self.tanh4(h11)
            h11 = self.linear5(h11)
            h11 = self.tanh5(h11)
            h11 = self.linear6(h11)
            h11 = self.tanh6(h11)
            h11 = self.linear7(h11)
            h11 = self.tanh7(h11)
            h11 = self.linear8(h11)
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
            h21 = self.tanh4(h21)
            h21 = self.linear5(h21)
            h21 = self.tanh5(h21)
            h21 = self.linear6(h21)
            h21 = self.tanh6(h21)
            h21 = self.linear7(h21)
            h21 = self.tanh7(h21)
            h21 = self.linear8(h21)
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


class gpnet2_1_5(torch.nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description: small with no sigmoid
        """
        super(gpnet2_1_5, self).__init__()
        self.linear1 = torch.nn.Linear(2, 10)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(10, 10)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(10, 10)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(10, 6)
        self.tanh4 = torch.nn.Tanh()
        self.linear5 = torch.nn.Linear(6, 1)

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
            h11 = self.tanh4(h11)
            h11 = self.linear5(h11)

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
            h21 = self.tanh4(h21)
            h21 = self.linear5(h21)

            h2 = h21
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


### nets of type 2-2-gp
class gpnet2_2_1(torch.nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description: pure GP
        """
        super(gpnet2_2_1, self).__init__()

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = x_train.clone()

        if x_test is not None:
            h2 = x_test.clone()

        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet2_2_2(torch.nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description: different nets: one  with sigmoid output
        """
        super(gpnet2_2_2, self).__init__()
        self.linear1 = torch.nn.Linear(2, 30)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(30, 30)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(30, 6)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(6, 1)
        self.sigm = torch.nn.Sigmoid()
        self.scale = torch.nn.Parameter(torch.Tensor([1.0]))

        self.linear21 = torch.nn.Linear(2, 30)
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


class gpnet2_2_3(torch.nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description: two outputs of same net, with sigmoid output
        """
        super(gpnet2_2_3, self).__init__()
        self.linear1 = torch.nn.Linear(2, 30)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(30, 20)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(20, 6)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(6, 2)
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
            h = self.scale * h11

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
            h2 = self.scale * h21

        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out
