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

    # the predict forward function
    def forward(self, x_train, y_train, x_test):
        # See the autograd section for explanation of what happens here.
        n = x_train.size(0)
        D = 0.5*(x_train - x_train.t()).pow(2)/self.lengthscale.pow(2)
        Kyy = self.sigma_f.pow(2)*torch.exp(-D) + self.sigma_n.pow(2) * torch.eye(n)

        # these are only built when predicting not when training
        with torch.no_grad():
            D = 0.5*(x_test - x_train.t()).pow(2)/self.lengthscale.pow(2)
            Kfy = self.sigma_f*torch.exp(-D)

            D = 0.5 * (x_test - x_test.t()).pow(2) / self.lengthscale.pow(2)
            Kff = self.sigma_f*torch.exp(-D)

            # solve
            v, LU = torch.gesv(y_train, Kyy)
            f_test = Kfy.mm(v)
            tmp, LU = torch.gesv(Kfy.t(), Kyy)
            tmp = torch.sum(Kfy * tmp.t(), dim=1)
            cov_f = self.sigma_f.pow(2) - tmp

        return f_test, cov_f

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'sigma_f={}, lengthscale={}, sigma_n={}'.format(
            self.sigma_f.item(), self.lengthscale.item(), self.sigma_n.item()
        )
