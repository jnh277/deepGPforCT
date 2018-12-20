import torch
import integration as int

# a quick example of using lambda functions
def myfunc(a, b):
    return a*b

a = 5
b = 10

print(myfunc(a, b))
func = lambda a: myfunc(a, b)

print(func(a))



##  a very basic integration example

def quadratic(x, a0, a1, a2):
    return a0 + a1*x + a2*x.pow(2)


a1 = torch.randn(1, requires_grad=True)
a0 = torch.ones(1, requires_grad=True)

func = lambda x: quadratic(x, a0, 2.0, 0.3)

a = -1
b = 1.5


simpsons = int.Simpsons()


Itrue = 4.1875
I = simpsons(func, a, b, 1e-6)       # n must be a multiple of 2


print(I.item())

I.backward()
print(a0.grad)





