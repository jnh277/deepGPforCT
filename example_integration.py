import torch
import integration as int

# # a quick example of using lambda functions
# def myfunc(a, b):
#     return a*b
#
# a = 5
# b = 10
#
# print(myfunc(a, b))
# func = lambda a: myfunc(a, b)
#
# print(func(a))



##  a very basic integration example

def quadratic(x, a0, a1, a2):
    return a0 + a1*x + a2*x.pow(2)


a1 = torch.randn(1, requires_grad=True)
a0 = torch.ones(1, requires_grad=True)

func = lambda x: quadratic(x, a0, 2.0, 0.3)
# func = torch.sin

a = -1
b = 1.5


simpsons = int.Simpsons(fcount_out=True, fcount_max=None, hmin=None)


# Itrue = 4.1875
# true  grad is 2.5
I, fcount= simpsons(func, a, b, 1e-6)       # n must be a multiple of 2

print(I.item())
print(fcount)
I.backward()
print(a0.grad)

a0.grad.zero_()
simpsons = int.Simpsons(fcount_out=False, fcount_max=20, hmin=None)     # limiting the  number of subdivisions
I = simpsons(func, a, b, 1e-6)       # n must be a multiple of 2
print(I.item())
I.backward()
print(a0.grad.item())

# 2D integration
a0 = torch.ones(1, requires_grad=True)
def quadratic_2D(x, y):
    return a0 + 0.8714*x + 0.7429*y + 0.6143*x*y + 0.4857*x.pow(2)+0.3571*y.pow(2) + 0.2286*x*y.pow(2) + 0.1*x.pow(2)*y

func = lambda x,y: quadratic_2D(x,y)

simpsons2D = int.Simpsons2D(fcount_out=True, fcount_max=3000)

I, fcount = simpsons2D(func, 0, 1, 0, 1, 1e-6)
# correctintegral is 2.2964
print(I)
print(fcount)
I.backward()
print(a0.grad.item())

func = lambda x,y: torch.sin(x) + torch.cos(y)

I, fcount = simpsons2D(func, 0, 1, 0, 5, 1e-6)
print(I)
print(fcount)
# correct integral is 1.301168678939781
