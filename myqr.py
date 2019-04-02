import torch

def myqr(tensor):
    qr = qr_wrapper.apply
    return qr(tensor)

class qr_wrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx is a context object that can be used to stash information
        # for backward computation
        q, r = torch.qr(input)
        ctx.q = q
        ctx.r = r
        return r

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        q = ctx.q
        r = ctx.r
        rplus = torch.pinverse(r)
        # print(rplus)
        tmp = torch.tril(r.mm(grad_output.t()) - grad_output.mm(r.t()), -1)
        # print(tmp)
        return q.mm(grad_output + tmp.mm(rplus.t()))
