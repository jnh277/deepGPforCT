import torch
import torch.autograd.gradcheck as gradcheck

class MulConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None

mulConstant = MulConstant.apply

constant = 5.0
A = torch.randn(3, 1, requires_grad=True)

B = mulConstant(A, constant)

C = sum(B)
C.backward()


print(A.grad)




class MyTimes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.A = A
        ctx.B = B
        return A*B

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.B, grad_output*ctx.A



myTimes = MyTimes.apply
A = torch.randn(3, 1, requires_grad=True)
B = torch.randn(3, 1, requires_grad=True)
C = myTimes(A, B)
D = sum(C)
D.backward()
print('Gradient with respect to A')
print(A.grad)
print('Gradient with respect to B')
print(B.grad)


class MyDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.A = A
        ctx.B = B
        return sum(A*B)

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.B, grad_output*ctx.A


myDot = MyDot.apply
input = (torch.randn(20,1,dtype=torch.double,requires_grad=True), torch.randn(20,1,dtype=torch.double,requires_grad=True))
test = gradcheck(myDot, input, eps=1e-6, atol=1e-4)
print('Gradient check')
print(test)



class MyLinSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b):
        # ctx is a context object that can be used to stash information
        # for backward computation
        Ainv = torch.inverse(A)

        out = Ainv.mm(b)
        ctx.Ainv = Ainv
        ctx.A = A
        ctx.b = b
        # ctx.C = C
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        A = ctx.A
        b = ctx.b
        Ainv = ctx.Ainv
        n = A.size(0)
        m = A.size(1)
        grad_A = torch.Tensor(n, m)
        grad_B = torch.Tensor(n, 1)
        for i in range(n):
            idd = torch.zeros(n, 1)
            idd[i,0] = 1.0
            grad_B[i,0] = sum(grad_output * Ainv.mm(idd))
            for j in range(m):
                idd = torch.zeros(n, m)
                idd[i, j] = 1.0
                grad_A[i, j] = sum(grad_output * (Ainv.mm(idd.mm(Ainv.mm(b)))))
        return grad_A, grad_B

myLinSolve = MyLinSolve.apply

A = torch.randn(3, 3,requires_grad=True)
# A = A.t().mm(A)
b = torch.randn(3, 1,requires_grad=True)

x = myLinSolve(A, b)
sum_x = sum(x)

sum_x.backward()
print(A.grad)
print(B.grad)

input = (torch.randn(3,3,dtype=torch.double,requires_grad=True), torch.randn(3,1,dtype=torch.double,requires_grad=True))
test = gradcheck(myDot, input, eps=1e-6, atol=1e-4)
print('Gradient check of MyLinSolve')
print(test)


class MySumLinSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b):
        # ctx is a context object that can be used to stash information
        # for backward computation
        Ainv = torch.inverse(A)

        out = sum(Ainv.mm(b))
        ctx.Ainv = Ainv
        ctx.A = A
        ctx.b = b
        # ctx.C = C
        return out

    @staticmethod
    def backward(ctx, grad_output_sum):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        A = ctx.A
        b = ctx.b
        Ainv = ctx.Ainv
        n = A.size(0)
        m = A.size(1)
        grad_output = grad_output_sum * torch.ones(n, 1)
        grad_A = torch.Tensor(n, m)
        grad_B = torch.Tensor(n, 1)
        for i in range(n):
            idd = torch.zeros(n, 1)
            idd[i,0] = 1.0
            grad_B[i,0] = sum(grad_output * Ainv.mm(idd))
            for j in range(m):
                idd = torch.zeros(n, m)
                idd[i, j] = 1.0
                grad_A[i, j] = sum(grad_output * (Ainv.mm(idd.mm(Ainv.mm(b)))))
        return grad_A, grad_B

mySumLinSolve = MySumLinSolve.apply

A2 = torch.randn(3, 3, requires_grad=True)
# A = A.t().mm(A)
b2 = torch.randn(3, 1, requires_grad=True)


sum_x = mySumLinSolve(A2, b2)
sum_x.backward()

input = (torch.randn(3,3,dtype=torch.double,requires_grad=True), torch.randn(3,1,dtype=torch.double,requires_grad=True))
test = gradcheck(myDot, input, eps=1e-6, atol=1e-4)
print('Gradient check of MySumLinSolve')
print(test)