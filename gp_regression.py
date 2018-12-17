import torch.nn as nn
import torch
import math


class GP_SE(nn.Module):
    def __init__(self, sigma_f, lengthscale, sigma_n):
        super(GP_SE, self).__init__()
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.sigma_f = nn.Parameter(torch.Tensor([sigma_f]))
        self.lengthscale = nn.Parameter(torch.as_tensor(lengthscale, dtype=torch.float))
        self.sigma_n = nn.Parameter(torch.Tensor([sigma_n]))
        self.training = True        # initially we want to be in training mode

    # the predict forward function
    def forward(self, x_train, y_train, x_test=None):
        # See the autograd section for explanation of what happens here.
        n = x_train.size(0)
        p = x_train.size(-1)
        d = torch.zeros(n, n)
        for i in range(p):
            d += 0.5*(x_train[:,i].unsqueeze(1) - x_train[:,i].unsqueeze(0)).pow(2)/self.lengthscale[i].pow(2)

        kyy = self.sigma_f.pow(2)*torch.exp(-d) + self.sigma_n.pow(2) * torch.eye(n)
        c = torch.cholesky(kyy, upper=True)
        # v = torch.potrs(y_train, c, upper=True)
        v, _ = torch.gesv(y_train, kyy)
        if x_test is None:
            out = (c, v)

        if x_test is not None:
            with torch.no_grad():
                ntest = x_test.size(0)
                d = torch.zeros(ntest, n)
                for i in range(p):
                    d += 0.5 * (x_test[:, i].unsqueeze(1) - x_train[:, i].unsqueeze(0)).pow(2) / self.lengthscale[i].pow(2)
                kfy = self.sigma_f.pow(2)*torch.exp(-d)
                # solve
                f_test = kfy.mm(v)
                tmp = torch.potrs(kfy.t(), c, upper=True)
                tmp = torch.sum(kfy * tmp.t(), dim=1)
                cov_f = self.sigma_f.pow(2) - tmp
            out = (f_test, cov_f)
        return out


    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'sigma_f={}, lengthscale={}, sigma_n={}'.format(
            self.sigma_f.item(), self.lengthscale.item(), self.sigma_n.item()
        )


class GP_1D(nn.Module):
    def __init__(self, sigma_f, lengthscale, sigma_n):
        super(GP_1D, self).__init__()
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.sigma_f = nn.Parameter(torch.Tensor([sigma_f]))
        self.lengthscale = nn.Parameter(torch.Tensor([lengthscale]))
        self.sigma_n = nn.Parameter(torch.Tensor([sigma_n]))
        self.training = True        # initially we want to be in training mode

    # the predict forward function
    def forward(self, x_train, y_train, x_test=None):
        # See the autograd section for explanation of what happens here.
        n = x_train.size(0)
        d = 0.5*(x_train - x_train.t()).pow(2)/self.lengthscale.pow(2)
        kyy = self.sigma_f.pow(2)*torch.exp(-d) + self.sigma_n.pow(2) * torch.eye(n)
        c = torch.cholesky(kyy, upper=True)
        # v = torch.potrs(y_train, c, upper=True)
        v, _ = torch.gesv(y_train, kyy)
        if x_test is None:
            out = (c, v)

        if x_test is not None:
            with torch.no_grad():
                d = 0.5*(x_test - x_train.t()).pow(2)/self.lengthscale.pow(2)
                kfy = self.sigma_f.pow(2)*torch.exp(-d)
                # solve
                f_test = kfy.mm(v)
                tmp = torch.potrs(kfy.t(), c, upper=True)
                tmp = torch.sum(kfy * tmp.t(), dim=1)
                cov_f = self.sigma_f.pow(2) - tmp
            out = (f_test, cov_f)
        return out


    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'sigma_f={}, lengthscale={}, sigma_n={}'.format(
            self.sigma_f.item(), self.lengthscale.item(), self.sigma_n.item()
        )


class NegMarginalLogLikelihood(nn.Module):
    def __init__(self):
        super(NegMarginalLogLikelihood, self).__init__()

    def forward(self, y, c, v):
        nLL = torch.dot(y, v.squeeze())/2 + torch.sum(torch.log(c.diag()))
        return nLL


class GP_SE_R(nn.Module):
    def __init__(self, sigma_f, lengthscale, sigma_n):
        super(GP_SE_R, self).__init__()
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.sigma_f = nn.Parameter(torch.Tensor([sigma_f]))
        self.lengthscale = nn.Parameter(torch.as_tensor(lengthscale, dtype=torch.float))
        self.sigma_n = nn.Parameter(torch.Tensor([sigma_n]))

    # the predict forward function
    def forward(self, x_train, y_train, body_train, x_test=None, body_test=None, classify=False):
        # See the autograd section for explanation of what happens here.
        n = x_train.size(0)
        p = x_train.size(-1)
        d = torch.zeros(n, n)

        nB = body_train.size(1)         # number of regions/bodies
        if classify:  # i.e we are predicting not training
            body_train = body_train > 0.5

        nullB = torch.sqrt(1-torch.sum(body_train.float().pow(2),1))

        for i in range(p):
            d += 0.5*(x_train[:,i].unsqueeze(1) - x_train[:,i].unsqueeze(0)).pow(2)/self.lengthscale[i].pow(2)

        kse = self.sigma_f.pow(2)*torch.exp(-d) + self.sigma_n.pow(2) * torch.eye(n)

        kyy = nullB.unsqueeze(0)*nullB.unsqueeze(1)*kse
        for i in range(nB):
            kyy += body_train[:,i].float().unsqueeze(1)*body_train[:,i].float().unsqueeze(0)*kse

        c = torch.cholesky(kyy, upper=True)
        # v = torch.potrs(y_train, c, upper=True)
        v, _ = torch.gesv(y_train, kyy)
        if x_test is None:
            out = (c, v)

        if x_test is not None:
            with torch.no_grad():
                if classify:
                    body_test = body_test > 0.5         # make a distinct classifier
                nullB_test = torch.sqrt(1 - torch.sum(body_test.float().pow(2), 1))
                ntest = x_test.size(0)
                d = torch.zeros(ntest, n)
                for i in range(p):
                    d += 0.5 * (x_test[:, i].unsqueeze(1) - x_train[:, i].unsqueeze(0)).pow(2) / self.lengthscale[i].pow(2)
                kse = self.sigma_f.pow(2)*torch.exp(-d)
                kfy = nullB_test.unsqueeze(1) * nullB.unsqueeze(0) * kse
                for i in range(nB):
                    kfy += body_test[:, i].float().unsqueeze(1) * body_train[:, i].float().unsqueeze(0) * kse

                # solve
                f_test = kfy.mm(v)
                tmp = torch.potrs(kfy.t(), c, upper=True)
                tmp = torch.sum(kfy * tmp.t(), dim=1)
                cov_f = self.sigma_f.pow(2) - tmp
            out = (f_test, cov_f)
        return out


    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'sigma_f={}, lengthscale={}, sigma_n={}'.format(
            self.sigma_f.item(), self.lengthscale.item(), self.sigma_n.item()
        )

class GP_1D_int(nn.Module):
    def __init__(self, sigma_f, lengthscale, sigma_n):
        super(GP_1D_int, self).__init__()
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.sigma_f = nn.Parameter(torch.Tensor([sigma_f]))
        self.lengthscale = nn.Parameter(torch.Tensor([lengthscale]))
        self.sigma_n = nn.Parameter(torch.Tensor([sigma_n]))
        self.training = True        # initially we want to be in training mode

    # the predict forward function
    def forward(self, x_train, y_train, x_test=None):
        # See the autograd section for explanation of what happens here.
        n = x_train.size(0)
        q1 = (x_train[:,1].view(n,1)-x_train[:,0].view(n,1).t()) / (2*self.lengthscale.pow(2)).sqrt()
        q2 = (x_train[:,1].view(n,1)-x_train[:,1].view(n,1).t()) / (2*self.lengthscale.pow(2)).sqrt()
        m1 = (x_train[:,0].view(n,1)-x_train[:,0].view(n,1).t()) / (2*self.lengthscale.pow(2)).sqrt()
        m2 = (x_train[:,0].view(n,1)-x_train[:,1].view(n,1).t()) / (2*self.lengthscale.pow(2)).sqrt()
        kyy = self.sigma_f.pow(2)*( self.lengthscale.pow(2)*math.sqrt(math.pi)*(  ( (q1*torch.erf(q1)+torch.exp(-q1.pow(2))/math.sqrt(math.pi)) -
        (q2*torch.erf(q2)+torch.exp(-q2.pow(2))/math.sqrt(math.pi)) ) + ((m2*torch.erf(m2)+torch.exp(-m2.pow(2))
        /math.sqrt(math.pi)) -(m1*torch.erf(m1)+torch.exp(-m1.pow(2))/math.sqrt(math.pi))) )  + self.sigma_n.pow(2)*torch.eye(n) )

        #d = 0.5*(x_train - x_train.t()).pow(2)/self.lengthscale.pow(2)
        #kyy = self.sigma_f.pow(2)*torch.exp(-d) + self.sigma_n.pow(2) * torch.eye(n)


        c = torch.cholesky(kyy, upper=True)
        # v = torch.potrs(y_train, c, upper=True)
        v, _ = torch.gesv(y_train, kyy) # kyy^-1 * y
        if x_test is None:
            out = (c, v)

        if x_test is not None:
            with torch.no_grad():
                kfy= ((math.sqrt(math.pi)/2)*(torch.erf((x_train[:,1].view(n,1)-x_test.t())/math.sqrt(2*self.lengthscale.pow(2)))
                -torch.erf((x_train[:,0].view(n,1)-x_test.t())/math.sqrt(2*self.lengthscale.pow(2))))*math.sqrt(2*self.lengthscale.pow(2)))
                kfy=self.sigma_f.pow(2)*kfy.t()
                # solve
                f_test = kfy.mm(v)
                tmp = torch.potrs(kfy.t(), c, upper=True)
                tmp = torch.sum(kfy * tmp.t(), dim=1)
                cov_f = self.sigma_f.pow(2) - tmp
            out = (f_test, cov_f)
        return out


    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'sigma_f={}, lengthscale={}, sigma_n={}'.format(
            self.sigma_f.item(), self.lengthscale.item(), self.sigma_n.item()
        )
