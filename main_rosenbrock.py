# -*- coding: utf-8 -*-
import optimization
import numpy as np
import scipy.linalg as sl
import scipy.optimize as opt
import matplotlib.pyplot as plt
'''TODO SIST = lägg till #fval m.m.'''
'''EV TODO = input function g into problem initialization.'''

rosenbrock = True

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
    

#problem = optimization.Problem(f, gradient_function=g)
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
#x0 = x0_options[8]
x0 = [-3.9, -1.1]
newton_method = newton_methods[1]
line_search_method = line_search_methods[2]
x, fmin, x_values, h_diff_values, h_quotient_values = \
    solver.find_local_min(newton_method, x0, line_search_method, debug=True)
#OBS. endast main.py använder h_diff och h_quotient.
    
if rosenbrock:
    
    #Plot the Rosenbrock-function
    def f(x,y):
        return 100*(y - x**2)**2 + (1 - x)**2

    #x = np.linspace(-5,15,1000)
    #y = np.linspace(-10, 150,1000)
    x = np.linspace(np.min(x_values[:][0])-10, np.max(x_values[:][0])+10,1000)
    y = np.linspace(np.min(x_values[:][1])-10, np.max(x_values[:][1])+20,1000)
    z = np.ndarray((len(x),len(y)))

    for i in range(0,len(x)):
        for j in range(0,len(y)):
            z[i][j] = f(x[i],y[j])
            
    contours = plt.contour(x,y,z.T,[5, 10,300,3000,3e4,3e5, 1e8])
    plt.clabel(contours,inline=1)
    
    #Plot the initial point
    plt.plot(x_values[0][0], x_values[0][1],  'bo', color = 'r')

    for i in range(len(x_values)-2):
        plt.plot(x_values[i+1][0], x_values[i+1][1],  'bo', color = 'b')
        
    #Plot the minimum point
    plt.plot(x_values[len(x_values)-1][0], x_values[i+1][1],  'bo', color = 'g')
    
    plt.show()
    #plt.close()