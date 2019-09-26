# -*- coding: utf-8 -*-

import numpy as np
import unittest
import optimization as op
import matplotlib.pyplot as plt
from timeit import default_timer as timer


'''
class test_classic_newton(unittest.TestCase):
    def setup(self):
'''

class test_gradient(unittest.TestCase):
    def f(x):
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    def setup(self):
        problem = op.Problem(f, 2)
        solver = op.Solver(problem)
        
        x = [4, 2]
        self.gradient = solver.compute_gradient(x)
        self.hessian = solver.compute_hessian(x)
        self.grad = np.array([100*(4*x[0]**3-4*x[0]*x[1])-2+2*x[0],100*(2*x[1]-2*x[0]**2)])
    def test(self):
        self.assertAlmostEqual(self.grad,self.gradient)
      