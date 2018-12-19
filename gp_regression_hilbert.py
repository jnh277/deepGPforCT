import torch.nn as nn
import torch
import math

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
        self.training = True  # initially we want to be in training mode [don't need this]

    # the predict forward function
    def forward(self, x_train, y_train, m, L, x_test=None):

        # phi_j(x) = 1/sqrt(L) * sin(pi*j*(x+L)/(2*L))
        # lambda_j = ( pi*j / (2L) )^2

        index = torch.empty(1, m)
        for i in range(m):
            index[0, i]=i+1

        # See the autograd section for explanation of what happens here.
        n = x_train.size(0)

        phi = ( 1/math.sqrt(L) ) * torch.sin(math.pi*index*(x_train+L)*0.5/L) # basis functions
        # diagonal of inverse lambda matrix
        inv_lambda_diag = ( self.sigma_f.pow(-2) * torch.pow(2*math.pi*self.lengthscale.pow(2), -0.5)*
                                  torch.exp( 0.5*self.lengthscale.pow(2)*pow(math.pi*index.t() / (2*L), 2) ) ).view(m)

        bigphi = phi.t().mm(phi) + self.sigma_n.pow(2) * torch.diag(inv_lambda_diag)

        c = torch.cholesky(bigphi, upper=True)
        v, _ = torch.gesv( phi.t().mm(y_train.view(n, 1)), bigphi)
        if x_test is None:
            out = (c, v, inv_lambda_diag, phi, self.sigma_n)

        if x_test is not None:
            with torch.no_grad():
                phi_star = 1/math.sqrt(L) * torch.sin(math.pi*index.t()*(x_test.t()+L)*0.5/L)

                # solve
                f_test = phi_star.t().mm(v)
                tmp = torch.potrs(phi_star, c, upper=True)
                cov_f = self.sigma_f.pow(2)*torch.sum(phi_star.t() * tmp.t(), dim=1)
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

    def forward(self, y, c, v, inv_lambda_diag, phi, sign):
        n=y.size(0)
        m=inv_lambda_diag.size(0)
        logQ = ( (n-m)*torch.log(sign.pow(2))
                + 2*torch.sum(torch.log(c.diag()))
                + torch.sum(torch.log(inv_lambda_diag)) )
        yQiy = sign.pow(-2)*( y.dot(y) - v.view(m).dot(y.view(1, n).mm(phi).view(m) ) )
        nLL = 0.5*logQ + 0.5*yQiy
        return nLL