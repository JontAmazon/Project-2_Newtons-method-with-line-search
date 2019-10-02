# -*- coding: utf-8 -*-
import time

import numpy as np
import scipy.linalg as sl
import scipy.optimize as opt
import matplotlib.pyplot as plt

import chebyquad_problem as cheb
import optimization
"""
    This script tests our optimizer on one of three functions:
        - 1D: 5 degree polynomial
        - 2D: Rosenbrock function
        - nD: Chebyquad problem
"""

current_milli_time = lambda: int(round(time.time() * 1000))

''' ROSENBROCK PROBLEM '''
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

''' 5 DEGREE POLYNOMIAL PROBLEM '''
def f1(x):
    return x**5 + 3.5*x**4 - 2.5*x**3 -12.5*x**2 + 1.5*x + 9
def g2(x):
    return 14*x**4 + 14*x**3 - 7.5*x**2 - 25*x + 1.5
def G2(x):
    return 20*x**3 + 42*x**2 - 15*x - 25



''' STANDARD INPUT CHOICES '''
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


''' MAIN PROGRAM '''
plt.close()
newton_method = newton_methods[4]
line_search_method = line_search_methods[2]
perform_plot = True
plot_pause_time = 0.05
cheby_bool = False
x0 = [4, 2]
#x0 = [0]


''' THE THREE DIFFERENT CASES BELOW '''
#        - nD: Chebyquad problem
#        - 1D: 5 degree polynomial
#        - 2D: Rosenbrock function

if cheby_bool==True:
    xmin= opt.fmin_bfgs(cheb.chebyquad,x0,cheb.gradchebyquad)  # should converge after 18 iterations  
    fmin = cheb.chebyquad(xmin)
    
 #   problem = optimization.Problem(chebyquad)
    problem = optimization.Problem(cheb.chebyquad, cheb.gradchebyquad)
    solver = optimization.Solver(problem, tol=1e-5, grad_tol=1e-6)
    our_xmin, our_fmin, x_values, useless1, useless2 = \
        solver.find_local_min(newton_methods[4], x0, line_search_methods[1])


if len(x0)==1 and not cheby_bool: #1D polynomial.
    problem = optimization.Problem(f1)
    solver = optimization.Solver(problem, tol=1e-5, grad_tol=1e-6)
    x, fmin, x_values, useless1, useless2 = \
        solver.find_local_min(newton_method, x0, line_search_method, debug=True)

    #Plot the 1D polynomial.
    #x = np.linspace(-3.2,1.8,500)
    if perform_plot:
        x = np.linspace(np.min(x_values)-0.5, np.max(x_values)+0.5, 1000)
        z = np.ndarray(len(x))
        for i in range(len(x)):
                z[i] = f1(x[i])
        z1 = np.ndarray(len(x_values))
        for i in range(len(x_values)):
                z1[i] = f1(x_values[i])
        plt.figure(1)
        plt.ion()   
        plt.plot(x, z)
                
        plt.plot(x_values[0], z1[0], 'bo', color='r')
        plt.title('Iteration: ' + str(0))
        plt.draw()
        plt.pause(1)
        for i in range(1,len(x_values)-2):
            plt.plot(x_values[i], z1[i], 'bo', color='b')
            plt.title('Iteration: ' + str(i))
            plt.draw()
            plt.pause(plot_pause_time)
        plt.plot(x_values[len(x_values)-1], z1[len(x_values)-1], 'bo', color='y')
        plt.title('Iteration: ' + str(len(x_values)-1))
        plt.draw()
        plt.ioff()
        plt.show()
    

elif len(x0)==2 and not cheby_bool: #2D Rosenbrock.
    problem = optimization.Problem(f)
    solver = optimization.Solver(problem, tol=1e-5, grad_tol=1e-6)
    x, fmin, x_values, h_diff_values, h_quotient_values = \
        solver.find_local_min(newton_method, x0, line_search_method, debug=True)

    """Task 12: I CHOSE to consider the Rosenbrock problem, and made two lists.
    The first one contains values for ||H - H_correct|| at each step.
    The second one contains values for ||H|| / ||H_correct||. Let's plot this.
    NOTE: Task 12 considers newton_methods[4], plus ANY line_search_method."""
    #plt.plot(range(len(h_diff_values)-1), h_diff_values[1:])
    #plt.figure()
    #plt.plot(range(len(h_quotient_values)-1), h_quotient_values[1:]) #

    if perform_plot:
        #(Reformat x_values: list of arrays of arrays --> list of arrays).
        x_values = [np.array([e[0][0], e[1][0]]) for e in x_values]

        #Plotting grid by linspace:        
        xx = [e[0] for e in x_values] #all x-values.
        yy = [e[1] for e in x_values] #all y-values.   
        xrange = np.max(xx) - np.min(xx)
        yrange = np.max(yy) - np.min(yy)        
        x = np.linspace(np.min(xx)-0.1*xrange, np.max(xx)+0.1*xrange, 1000)
        y = np.linspace(np.min(yy)-0.1*yrange, np.max(yy)+0.1*yrange, 1000)
        
        #Plot countours (z==constant) using the
        def f(x,y): #Rosenbrock function.
            return 100*(y - x**2)**2 + (1 - x)**2
        
        z = np.ndarray((len(x),len(y)))
        for i in range(0,len(x)):
            for j in range(0,len(y)):
                z[i][j] = f(x[i],y[j])    
                
        plt.figure(1)
        plt.ion()   
        plt.xlim((np.min(x), np.max(x)))
        plt.ylim((np.min(y), np.max(y)))
        contours = plt.contour(x,y,z.T,[1, 5, 1e2, 1e4, 3e4, 1e8])
        plt.clabel(contours,inline=1)
        plt.plot(x_values[0][0],x_values[0][0],'bo',color='r',markersize=4) 
        #Plot all steps.
        for i in range(1,len(x_values)-2):
            plt.plot(x_values[i][0],x_values[i][0],'bo',color='b',markersize=4)
            plt.title('Iteration: ' + str(i))
            plt.draw()
            plt.pause(plot_pause_time)
        plt.plot(x_values[-1][0],x_values[-1][1],'bo',color='y',markersize=6)
        plt.ioff()
        plt.show()
        # plt.plot(x_values[0][0], x_values[0][1],  'bo', color = 'r')
        # for i in range(len(x_values)-2):
        #     plt.plot(x_values[i+1][0], x_values[i+1][1],  'bo', color = 'b')        
        # plt.plot(x_values[len(x_values)-1][0], x_values[i+1][1],  'bo', color = 'g')
        # plt.show()

if cheby_bool==True:
    x0=np.linspace(0,1,4)
    time1 = current_milli_time()
    xmin= opt.fmin_bfgs(cheb.chebyquad,x0,cheb.gradchebyquad)  # should converge after 18 iterations
    print('Required time: ' + str(current_milli_time()-time1) + 'ms')
    fmin = cheb.chebyquad(xmin)
    print('xmin ' + str(xmin))
    print('fmin ' + str(fmin))
    
        # Our solver:
 #   problem = optimization.Problem(chebyquad)
    problem = optimization.Problem(cheb.chebyquad, cheb.gradchebyquad)
    solver = optimization.Solver(problem, tol=1e-6, grad_tol=1e-6)
    time2 = current_milli_time()
    our_xmin, our_fmin, x_values, useless1, useless2 = \
        solver.find_local_min(newton_methods[4], x0, line_search_methods[2])
    print('Required time: ' + str(current_milli_time()-time2) + 'ms')
















