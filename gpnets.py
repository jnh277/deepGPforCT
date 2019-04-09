import torch
import torch.nn as nn
import gp_regression_hilbert as gprh

# get the number of (trainable) parameters in model
def numel(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

########################################################################################################################
##################################### 1D input #########################################################################
########################################################################################################################


########################################################################################################################
### nets of type 1-1-gp ################################################################################################
########################################################################################################################
class gpnet1_1_1(nn.Module):
    def __init__(self, sigma_f=1.0, lengthscale=1.0, sigma_n=1.0):
        """
        Description: pure GP
        """
        super(gpnet1_1_1, self).__init__()

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=[lengthscale], sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = True

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


class gpnet1_1_2(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=1, sigma_n=1):
        """
        Description:
        """
        super(gpnet1_1_2, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(1, 20),
        nn.Tanh(),
        nn.Linear(20, 8),
        nn.Tanh(),
        nn.Linear(8, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
        # nn.Sigmoid(),
        # nn.Linear(1, 1)
        )
        # self.scale  = torch.nn.Parameter(torch.Tensor([1.0]))
        # self.scale2 = torch.nn.Parameter(torch.Tensor([1.0]))

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=[lengthscale], sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h =  self.mynet( x_train.clone() ) .add( x_train.clone() )

        if x_test is not None:
            h2 =  self.mynet( x_test.clone() ) .add( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet1_1_3(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=1, sigma_n=1):
        """
        Description:
        """
        super(gpnet1_1_3, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(1, 12),
        nn.Tanh(),
        nn.Linear(12, 6),
        nn.Sigmoid(),
        nn.Linear(6, 1),
        # nn.Hardtanh()
        # nn.Linear(1, 1, bias=False)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=[lengthscale], sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out

class gpnet1_1_4(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=1, sigma_n=1):
        """
        Description:
        """
        super(gpnet1_1_4, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(1, 6),
        nn.Tanh(),
        nn.Linear(6, 3),
        nn.Tanh(),
        nn.Linear(3, 1),
        nn.Sigmoid(),
        nn.Linear(1, 1, bias=False)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=[lengthscale], sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out

########################################################################################################################
### nets of type 1-2-gp ################################################################################################
########################################################################################################################
class gpnet1_2_1(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description:
        """
        super(gpnet1_2_1, self).__init__()
        self.mynet1 = nn.Sequential(
        nn.Linear(1, 30),
        nn.Tanh(),
        nn.Linear(30, 30),
        nn.Tanh(),
        nn.Linear(30, 6),
        nn.Tanh(),
        nn.Linear(6, 1),
        nn.Sigmoid(),
        nn.Linear(1, 1, bias=False)
        )

        self.mynet2 = nn.Sequential(
        nn.Linear(1, 30),
        nn.Tanh(),
        nn.Linear(30, 6),
        nn.Tanh(),
        nn.Linear(6, 1)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h11 = self.mynet1( x_train.clone() )
            h12 = self.mynet2( x_train.clone() )

            h = torch.cat((h11,h12),1)

        if x_test is not None:
            h21 = self.mynet1( x_test.clone() )
            h22 = self.mynet2( x_test.clone() )

            h = torch.cat((h21,h22),1)
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet1_2_2(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description:
        """
        super(gpnet1_2_2, self).__init__()
        self.mynet1 = nn.Sequential(
        nn.Linear(1, 10),
        nn.Tanh(),
        nn.Linear(10, 5),
        nn.Tanh(),
        nn.Linear(5, 1),
        nn.Sigmoid(),
        nn.Linear(1, 1, bias=False)
        )

        self.mynet2 = nn.Sequential(
        nn.Linear(1, 10),
        nn.Tanh(),
        nn.Linear(10, 5),
        nn.Tanh(),
        nn.Linear(5, 1)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h11 = self.mynet1( x_train.clone() )
            h12 = self.mynet2( x_train.clone() )

            h = torch.cat((h11,h12),1)

        if x_test is not None:
            h21 = self.mynet1( x_test.clone() )
            h22 = self.mynet2( x_test.clone() )

            h = torch.cat((h21,h22),1)
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


########################################################################################################################
##################################### 2D input #########################################################################
########################################################################################################################


########################################################################################################################
### nets of type 2-1-gp ################################################################################################
########################################################################################################################
class gpnet2_1_1(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description:
        """
        super(gpnet2_1_1, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 30),
        nn.Tanh(),
        nn.Linear(30, 30),
        nn.Tanh(),
        nn.Linear(30, 6),
        nn.Tanh(),
        nn.Linear(6, 1),
        nn.Sigmoid(),
        nn.Linear(1, 1, bias=False)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet2_1_2(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description: smaller than above
        """
        super(gpnet2_1_2, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 40),
        nn.BatchNorm1d(40),
        nn.Tanh(),
        nn.Linear(40, 6),
        nn.BatchNorm1d(6),
        nn.Tanh(),
        nn.Linear(6, 1),
        # nn.Sigmoid()
        # nn.Linear(1, 1, bias=False)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet2_1_3(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description:
        """
        super(gpnet2_1_3, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 80),
        nn.Tanh(),
        nn.Linear(80, 80),
        nn.Tanh(),
        nn.Linear(80, 80),
        nn.Tanh(),
        nn.Linear(80, 80),
        nn.Tanh(),
        nn.Linear(80, 1),
        nn.Sigmoid(),
        nn.Linear(1, 1, bias=False)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet2_1_4(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description: deeper than the previous net
        """
        super(gpnet2_1_4, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 90),
        nn.Tanh(),
        nn.Linear(90, 90),
        nn.Tanh(),
        nn.Linear(90, 90),
        nn.Tanh(),
        nn.Linear(90, 90),
        nn.Tanh(),
        nn.Linear(90, 90),
        nn.Tanh(),
        nn.Linear(90, 90),
        nn.Tanh(),
        nn.Linear(90, 90),
        nn.Tanh(),
        nn.Linear(90, 90),
        nn.Tanh(),
        nn.Linear(90, 90),
        nn.Tanh(),
        nn.Linear(90, 90),
        nn.Tanh(),
        nn.Linear(90, 90),
        nn.Tanh(),
        nn.Linear(90, 1)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet2_1_5(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description: bit less neurons than previous
        """
        super(gpnet2_1_5, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 30),
        nn.Tanh(),
        nn.Linear(30, 30),
        nn.Tanh(),
        nn.Linear(30, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 30),
        nn.Tanh(),
        nn.Linear(30, 30),
        nn.Tanh(),
        nn.Linear(30, 30),
        nn.Tanh(),
        nn.Linear(30, 1),
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet2_1_6(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description: small with no sigmoid
        """
        super(gpnet2_1_6, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 6),
        nn.Tanh(),
        nn.Linear(6, 1),
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet2_1_7(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description: exponential decay of neurons
        """
        super(gpnet2_1_7, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 300),
        nn.Tanh(),
        nn.Linear(300, 150),
        nn.Tanh(),
        nn.Linear(150, 75),
        nn.Tanh(),
        nn.Linear(75, 30),
        nn.Tanh(),
        nn.Linear(30, 1)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out

class gpnet2_1_8(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description: fewer neurons than previous
        """
        super(gpnet2_1_8, self).__init__()
        # self.mynet = nn.Sequential(
        # nn.Linear(2, 10),
        # # nn.BatchNorm1d(10),
        # nn.Tanh(),
        # nn.Linear(10, 10),
        # # nn.BatchNorm1d(10),
        # nn.Tanh(),
        # nn.Linear(10, 5),
        # # nn.BatchNorm1d(5),
        # nn.Tanh(),
        # nn.Linear(5, 2),
        # # nn.BatchNorm1d(2),
        # nn.Tanh(),
        # nn.Linear(2, 1),
        # # nn.Sigmoid(),
        # # nn.Linear(1, 1, bias=False)
        # )
        self.mynet = nn.Sequential(
        nn.Linear(2, 6),
        # nn.BatchNorm1d(10),
        nn.Tanh(),
        # nn.Linear(20, 10),
        # nn.BatchNorm1d(10),
        # nn.Tanh(),
        # nn.Linear(10, 5),
        # nn.BatchNorm1d(5),
        # nn.Tanh(),
        # nn.Linear(5, 2),
        # nn.BatchNorm1d(2),
        nn.Tanh(),
        nn.Linear(6, 1),
        nn.Sigmoid(),
        # nn.Linear(1, 1, bias=False)
        )
        # self.scale  = torch.nn.Parameter(torch.Tensor([10.0]))
        # self.scale2 = torch.nn.Parameter(torch.Tensor([10.0]))

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )
            # h =  self.scale/(1.0+torch.exp(-(h-0.5).mul(self.scale2)))    .add(    x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
            # h2 =  self.scale/(1.0+torch.exp(-(h2-0.5).mul(self.scale2)))    .add(    x_test.clone() )

        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out

class gpnet2_1_9(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1], sigma_n=1):
        """
        Description: larger than previous
        """
        super(gpnet2_1_9, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 500),
        nn.Tanh(),
        nn.Linear(500, 400),
        nn.Tanh(),
        nn.Linear(400, 320),
        nn.Tanh(),
        nn.Linear(320, 256),
        nn.Tanh(),
        nn.Linear(256, 204),
        nn.Tanh(),
        nn.Linear(204, 162),
        nn.Tanh(),
        nn.Linear(162, 131),
        nn.Tanh(),
        nn.Linear(131, 104),
        nn.Tanh(),
        nn.Linear(104, 83),
        nn.Tanh(),
        nn.Linear(83, 66),
        nn.Tanh(),
        nn.Linear(66, 40),
        nn.Tanh(),
        nn.Linear(40, 1)
        )
        
        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out

########################################################################################################################
### nets of type 2-2-gp ################################################################################################
########################################################################################################################
class gpnet2_2_1(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description: pure GP
        """
        super(gpnet2_2_1, self).__init__()

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = True

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


class gpnet2_2_2(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description: different nets: one  with sigmoid output
        """
        super(gpnet2_2_2, self).__init__()
        self.mynet1 = nn.Sequential(
        nn.Linear(2, 20),
        nn.Tanh(),
        nn.Linear(20, 10),
        nn.Tanh(),
        nn.Linear(10, 5),
        nn.Tanh(),
        nn.Linear(5, 1),
        nn.Sigmoid(),
        nn.Linear(1, 1, bias=False)
        )

        self.mynet2 = nn.Sequential(
        nn.Linear(2, 20),
        nn.Tanh(),
        nn.Linear(20, 10),
        nn.Tanh(),
        nn.Linear(10, 1),
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h11 = self.mynet1( x_train.clone() )
            h12 = self.mynet2( x_train.clone() )

            h = torch.cat((h11,h12),1)

        if x_test is not None:
            h21 = self.mynet1( x_test.clone() )
            h22 = self.mynet2( x_test.clone() )

            h2 = torch.cat((h21,h22),1)
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet2_2_3(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description: two outputs of same net, with sigmoid output
        """
        super(gpnet2_2_3, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 30),
        nn.Tanh(),
        nn.Linear(30, 20),
        nn.Tanh(),
        nn.Linear(20, 6),
        nn.Tanh(),
        nn.Linear(6, 2),
        nn.Sigmoid(),
        nn.Linear(2, 2)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() ) .add( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() ) .add( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet2_2_4(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description: two outputs of same net, no sigmoid outputs
        """
        super(gpnet2_2_4, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 20),
        nn.Tanh(),
        nn.Linear(20, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 20),
        nn.Tanh(),
        nn.Linear(20, 2),
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet2_2_5(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description: as net 2_2_4, but fewer neurons
        """
        super(gpnet2_2_5, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 20),
        nn.Tanh(),
        nn.Linear(20, 20),
        nn.Tanh(),
        nn.Linear(20, 30),
        nn.Tanh(),
        nn.Linear(30, 30),
        nn.Tanh(),
        nn.Linear(30, 20),
        nn.Tanh(),
        nn.Linear(20, 20),
        nn.Tanh(),
        nn.Linear(20, 20),
        nn.Tanh(),
        nn.Linear(20, 2)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h2 = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out


class gpnet2_2_6(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1], sigma_n=1):
        """
        Description: as net 2_2_4, but more neurons
        """
        super(gpnet2_2_6, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 20),
        nn.Tanh(),
        nn.Linear(20, 70),
        nn.Tanh(),
        nn.Linear(70, 70),
        nn.Tanh(),
        nn.Linear(70, 70),
        nn.Tanh(),
        nn.Linear(70, 70),
        nn.Tanh(),
        nn.Linear(70, 70),
        nn.Tanh(),
        nn.Linear(70, 20),
        nn.Tanh(),
        nn.Linear(20, 2)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out

########################################################################################################################
### nets of type 2-3-gp ################################################################################################
########################################################################################################################
class gpnet2_3_1(nn.Module):
    def __init__(self, sigma_f=1, lengthscale=[1,1,1], sigma_n=1):
        """
        Description: three outputs of same net, no sigmoid outputs
        """
        super(gpnet2_3_1, self).__init__()
        self.mynet = nn.Sequential(
        nn.Linear(2, 20),
        nn.Tanh(),
        nn.Linear(20, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 20),
        nn.Tanh(),
        nn.Linear(20, 3)
        )

        self.gp = gprh.GP_new(sigma_f=sigma_f, lengthscale=lengthscale, sigma_n=sigma_n)

        self.npar = numel(self)
        self.pureGP = False

    def forward(self, x_train=None, y_train=None, phi=None, sq_lambda=None, L=None, x_test=None):
        if x_train is not None:
            h = self.mynet( x_train.clone() )

        if x_test is not None:
            h = self.mynet( x_test.clone() )
        else:
            h2 = None
        if y_train is not None:
            out = self.gp(y_train,phi,sq_lambda,L,h2)
        else:
            out = h
        return out
