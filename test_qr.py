import torch
import torch.autograd.gradcheck as gradcheck

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
        tmp = torch.tril(r.mm(grad_output.t()) - grad_output.mm(r.t()), -1)
        return q.mm(grad_output + tmp.mm(rplus))




qr = qr_wrapper.apply

a = torch.ones(1,1,requires_grad=True)
rs = qr(a)
rs.backward()
print(a.grad) # this should be 1 which it is

w = torch.ones(1,1, requires_grad=True)
A3 = torch.ones(3,3, requires_grad=False)
A3[0, 0] = w
r3 = qr(A3)
v3 = torch.sum(r3.diag().abs())
v3.backward()
print(w.grad)  # this should equal 1.3938 which it does




# v.backward()
