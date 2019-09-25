# -*- coding: utf-8 -*-
import optimization



def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2


problem = optimization.Problem(f, 2)
solver = optimization.Solver(problem)

x = [4, 2]
gradient = solver.compute_gradient(x)
hessian = solver.compute_hessian(x)


