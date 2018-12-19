import torch.nn as nn
import torch

class Simpsons(nn.Module):
    def __init__(self):
        super(Simpsons, self).__init__()

    def forward(self, method, a, b, n):
        # n must be a multiple of 2
        n = round(n/2)*2
        h = (b - a)/n
        tau = torch.linspace(a, b, n+1)
        w = torch.Tensor(n+1)
        w[1::2] = 2.0
        w[::2] = 4.0
        w[0] = 1.0
        w[n] = 1.0

        f = method(tau)
        return sum(f * w)*h/3


class Simpsons2(nn.Module):
    def __init__(self):
        super(Simpsons2, self).__init__()

    def forward(self, method, a, b, n):
        n = round(n/3)*3
        h = (b - a)/n
        tau = torch.linspace(a, b, n+1)
        w = 24*torch.ones(n+1)
        # w[3:n] = 24
        w[0] = 9.0; w[1] = 28.0; w[2] = 23.0+24.0
        w[n] = 9.0; w[n-1] = 28.0; w[n-2] = 23.0

        f = method(tau)
        return sum(f * w)*h/24












