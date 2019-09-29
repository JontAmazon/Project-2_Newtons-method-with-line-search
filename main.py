# -*- coding: utf-8 -*-
import optimization
import numpy as np
import scipy.linalg as sl
import scipy.optimize as opt
'''EV TODO = input function g into problem initialization.'''


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
solver = optimization.Solver(problem, dimensions=2, max_iterations=1000, tol=1e-5, grad_tol=1e-6, hess_tol=1e-3)
newton_methods = ['exact_newton', 'good_broyden', 'bad_broyden', \
                  'davidon_fletcher_powell', 'broyden_fletcher_goldfarb_shanno']
line_search_methods = [None, 'exact_line_search', 'wolfe-powell', 'goldstein']
x0_options = [[1, 1],    #0
              [1, 1.1],  #1
              [1, 2],    #2
              [1, 10],   #3
              [1, 100],  #4
              [1, 1000], #5
              [1.1, 1],  #6
              [2, 1],    #7
              [10, 1],   #8
              [100, 1],  #9
              [1000, 1], #10
              [1.1, 1.1], #11
              [2, 2],    #12
              [10, 10],  #13
              [100, 100],#14
              [1000,1000]]#15
x0 = x0_options[13]
newton_method = newton_methods[0]
line_search_method = line_search_methods[0]
x, fmin = solver.find_local_min(newton_method, x0, line_search_method, debug=True)







''' KAN VA BRA ATT HA: '''
# För att testa våra funktioner för att beräkna grad och hess:
#print(solver.compute_gradient(x0) / g(x0))  
#print(solver.compute_hessian(x0) / G(x0))

#xx, ffmin = opt.fmin_bfgs(f, np.array(x0))
#optimum = opt.minimize(f, np.array(x0))






