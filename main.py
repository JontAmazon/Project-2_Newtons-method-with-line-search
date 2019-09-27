# -*- coding: utf-8 -*-
import optimization
import numpy as np
import scipy.linalg as sl
import scipy.optimize as opt
'''EV TODO = calculate gradient and hessian with numpy!'''


def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def g(x):
    grad = np.zeros((2,1))
    grad[0,0] = 400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2
    grad[1,0] = -200*x[0]**2 + 200*x[1]
    return grad

def G(x):
    hess = np.zeros((2,2))
    hess[0,0] = 1200*x[0]**2 - 400*x[1] + 2
    hess[0,1] = hess[1,0] = -400*x[0]
    hess[1,1] = 200
    return hess
    

problem = optimization.Problem(f)
solver = optimization.Solver(problem, dimensions=2, tol=1e-5, grad_tol=1e-6, hess_tol=1e-3)

newton_methods = ['exact_newton', 'good_broyden', 'bad_broyden', \
                  'davidon_fletcher_powell', 'broyden_fletcher_goldfarb_shanno']
line_search_methods = [None, 'exact_line_search', 'wolfe-powell', 'goldstein']

# x, nice grad_tol/hess_tol
#x0 = [4, 2]    #1e-8   1e-3
x0 = [1.0, 1000.0]    #1e-8   1e-5
#x0 = [10, 10]
#x0 = [37, 100] #1e-6   1e-3
x, fmin = solver.find_local_min(newton_methods[0], x0, line_search_methods[0], debug=True)


#blip = solver.compute_gradient(x0)
#blop = g(x0)
#print(solver.compute_gradient(x0) / g(x0))  
#print(solver.compute_hessian(x0) / G(x0))



#x, fmin = solver.find_local_min(newton_methods[0], x0, line_search_methods[0], debug=True)
#xx, ffmin = opt.fmin_bfgs(f, np.array(x0))
#optimum = opt.minimize(f, np.array(x0))
#g = solver.compute_gradient(x0)
#G = solver.compute_hessian(x0)
#H = sl.inv(G)






