import torch.nn as nn
import torch


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
