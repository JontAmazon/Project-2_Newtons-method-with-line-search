# -*- coding: utf-8 -*-

import numpy as np
import unittest
import optimization as op
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import scipy.linalg as lg


'''
class test_classic_newton(unittest.TestCase):
    def setup(self):
'''

class test_gradient(unittest.TestCase):
    
    def setUp(self):
        x = np.array([1., 1.])
        def f(x):
            return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
        problem = op.Problem(f)
        solver = op.Solver(problem, tol=1e-5, grad_tol=1e-4, hess_tol=1e-5)
        newton_methods = ['exact_newton', 'good_broyden', 'bad_broyden', \
                          'davidon_fletcher_powell', 'broyden_fletcher_goldfarb_shanno']
        line_search_methods = [None, 'exact_line_search', 'wolfe-powell', 'goldstein']
        x, fmin = solver.find_local_min(newton_methods[0], x, line_search_methods[0], debug=True)
        
        
        self.gradient = solver.compute_gradient(x)
        g1 = self.gradient
        self.hessian = solver.compute_hessian(x)
        self.grad = np.array([100*(4*x[0]**3-4*x[0]*x[1])-2+2*x[0],100*(2*x[1]-2*x[0]**2)])
        g2= self.grad
        diff = lg.norm(self.gradient-self.grad)/lg.norm(self.grad)

        self.diff = diff
    def test_dimension(self):
        self.assertEqual(np.shape(self.gradient),(2,1))
    def test_correct_value(self):
        
        self.assertAlmostEqual(self.diff,0,1) #'not almost equal'
        
        

        
if __name__ =='__main__':
    unittest.main()