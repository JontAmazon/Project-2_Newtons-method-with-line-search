# -*- coding: utf-8 -*-
import optimization
import numpy as np
import scipy.linalg as sl
import scipy.optimize as opt


def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def g(x):
    '''TODO = testa om det funkar också med gradient som input.'''
    pass


problem = optimization.Problem(f)
solver = optimization.Solver(problem)

newton_methods = ['exact_newton', 'good_broyden', 'bad_broyden', \
                  'davidson_fletcher_powell', 'broyden_fletcher_goldfarb_shanno']
line_search_methods = [None, 'exact_line_search', 'wolfe-powell', 'goldstein']

x0 = [4, 2]
xx, ffmin = opt.fmin_bfgs(f, x0)
#x, fmin = solver.find_local_min(newton_methods[0], x0, line_search_methods[0], debug=True)






'''TODO = kolla open source exact newton method... '''



#g = solver.compute_gradient(x0)
#G = solver.compute_hessian(x0)
#H = sl.inv(G)








