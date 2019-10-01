# -*- coding: utf-8 -*-
import optimization
import numpy as np
import scipy.linalg as sl
import scipy.optimize as opt
import matplotlib.pyplot as plt
import chebyquad_problem as cheb
"""
    This script tests our optimizer on one of three functions:
        - 1D: 5 degree polynomial
        - 2D: Rosenbrock function
        - nD: Chebyquad problem
"""


''' ROSENBROCK PROBLEM '''
def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2                              # TODO = detta kan nog importas istället.
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

''' 5 DEGREE POLYNOMIAL PROBLEM '''
def f1(x):
    return x**5 + 3.5*x**4 - 2.5*x**3 -12.5*x**2 + 1.5*x + 9
def g2(x):
    return 14*x**4 + 14*x**3 - 7.5*x**2 - 25*x + 1.5
def G2(x):
    return 20*x**3 + 42*x**2 - 15*x - 25



''' MAIN PROGRAM '''
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
              [1000,1000],#15
              [-12, 1],   #16 "känd svår punkt"
              [-5, -5],   #17
              [-10, -10]] #18

newton_method = newton_methods[4]
line_search_method = line_search_methods[2]

cheby_bool = True
#x0 = [.5]
#x0 = x0_options[13]
x0 = np.linspace(0,1,4) #OK?


if len(x0)==1 and not cheby_bool:
    problem = optimization.Problem(f1)
    solver = optimization.Solver(problem, dimensions=1, max_iterations=1000, \
                                 tol=1e-5, grad_tol=1e-6, hess_tol=1e-3)
    x, fmin, x_values, useless1, useless2 = \
        solver.find_local_min(newton_method, x0, line_search_method, debug=True)

    #Plot the 1D polynomial.
    #x = np.linspace(-3.2,1.8,500)
    x = np.linspace(np.min(x_values)-0.5, np.max(x_values)+0.5, 1000)
    z = np.ndarray(len(x))
    for i in range(len(x)):
            z[i] = f1(x[i])
    z1 = np.ndarray(len(x_values))
    for i in range(len(x_values)):
            z1[i] = f1(x_values[i])
    plt.plot(x, z)        
    plt.plot(x_values[0], z1[0], 'bo', color='r')
    for i in range(1,len(x_values)-2):
        plt.plot(x_values[i], z1[i], 'bo', color='b')
    plt.plot(x_values[len(x_values)-1], z1[len(x_values)-1], 'bo', color='g')
    plt.show()
    #plt.close()
    

elif len(x0)==2 and not cheby_bool:
    problem = optimization.Problem(f)
    solver = optimization.Solver(problem, dimensions=2, max_iterations=1000, \
                                 tol=1e-5, grad_tol=1e-6, hess_tol=1e-3)
    x, fmin, x_values, h_diff_values, h_quotient_values = \
        solver.find_local_min(newton_method, x0, line_search_method, debug=True)

    """Task 12: I CHOSE to consider the Rosenberg problem, and made two lists.
    The first one contains values for ||H - H_correct|| at each step.
    The second one contains values for ||H|| / ||H_correct||. Let's plot this.
    NOTE: Task 12 considers newton_methods[4], plus ANY line_search_method."""
    #plt.plot(range(len(h_diff_values)-1), h_diff_values[1:])
    #plt.figure()
    #plt.plot(range(len(h_quotient_values)-1), h_quotient_values[1:]) #

    """PLOT"""
    #Plot countours.
    x = np.linspace(np.min(x_values[:][0])-10, np.max(x_values[:][0])+10,1000)
    y = np.linspace(np.min(x_values[:][1])-10, np.max(x_values[:][1])+20,1000)
    z = np.ndarray((len(x),len(y)))
    def f(x,y):
        return 100*(y - x**2)**2 + (1 - x)**2
    for i in range(0,len(x)):
        for j in range(0,len(y)):
            z[i][j] = f(x[i],y[j])            
    contours = plt.contour(x,y,z.T,[5, 10,300,3000,3e4,3e5, 1e8])
    plt.clabel(contours,inline=1)
    
    #Plot all steps.
    plt.plot(x_values[0][0], x_values[0][1],  'bo', color = 'r')
    for i in range(len(x_values)-2):
        plt.plot(x_values[i+1][0], x_values[i+1][1],  'bo', color = 'b')        
    plt.plot(x_values[len(x_values)-1][0], x_values[i+1][1],  'bo', color = 'g')
    plt.show()

if cheby_bool==True:
    xmin= opt.fmin_bfgs(cheb.chebyquad,x0,cheb.gradchebyquad)  # should converge after 18 iterations  
    fmin = cheb.chebyquad(xmin)
    
 #   problem = optimization.Problem(chebyquad)
    problem = optimization.Problem(cheb.chebyquad, cheb.gradchebyquad)
    solver = optimization.Solver(problem, max_iterations=1000,dimensions=len(x), tol=1e-5, grad_tol=1e-6, hess_tol=1e-3)
    newton_methods = ['exact_newton', 'good_broyden', 'bad_broyden', \
                  'davidon_fletcher_powell', 'broyden_fletcher_goldfarb_shanno']
    line_search_methods = [None, 'exact_line_search', 'wolfe-powell', 'goldstein']
    our_xmin, our_fmin, x_values, useless1, useless2 = \
        solver.find_local_min(newton_methods[4], x0, line_search_methods[1])
















