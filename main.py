# -*- coding: utf-8 -*-
import optimization

def f(x):
    return 100*(x[1] - x[0]^2)^2 + (1 - x[0])^2

optimizer = Optimization(f, 2)

x1 = [0, 0]
x2 = [4, 0]
x3 = [0, 4]
x4 = [4, 4]
hessian1 = optimizer.compute_hessian(x1)
hessian2 = optimizer.compute_hessian(x2)
hessian3 = optimizer.compute_hessian(x3)
hessian4 = optimizer.compute_hessian(x4)
