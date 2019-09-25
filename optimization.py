'''
    Script containing an optimization problem class, which contains methods for
    solving optimization problems using different methods
                                                                            '''
import numpy as np
import scipy

class Problem(object):
    def __init__(self, objective_function, dimensions, gradient=None):
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.gradient = gradient #(might be equal to None).
        

class Solver(object):
    def __init__(self, problem, g_tolerance=1e-6, g_delta=1e-4):
        self.objective_function = problem.objective_function
        self.dimensions = problem.dimensions
        self.gradient_function = problem.gradient #(might be equal to None).
        self.g_tolerance = g_tolerance
        self.g_delta = g_delta
        
        #todo: beräkna G och sedan H^0 = G^-1


    def find_local_min(self, newton_method, line_search_method):
        """Solves the problem of finding a local minimum of the function 
            described in the input problem, using a Pseudo-Newton method
            together with line search.
        """
        pass
    
    def compute_gradient(self, x):
        # Do we have an explicit function for the gradient? Then use it!
        if self.gradient_function != None:
            return self.gradient_function(x)
        
        # If not, compute it with finite differences:
        #   g = (f(x+dx) - f(x))/dx
        n = self.dimensions
        gradient = np.zeros(n)
        f = self.objective_function
        fx = f(x) #we only need to calculate this once
        for i in range(n):
            xx = x.copy()
            xx[i] = xx[i] + self.g_delta
            gradient[i] = (f(xx) - fx) / self.g_delta
        return gradient
    
    def compute_hessian(self, x):
        # The i:th column of the Hessian G_i equals g(x) differentiated w.r.t. x_i
        # This is approximated with a finite difference:
        # G_i = (g(x + tol*e_i) - g(x))/tol
        n = self.dimensions
        hessian = np.zeros((n,n))
        g = self.compute_gradient
        gx = g(x) #we only need to calculate this once
        for i in range(n):
            xx = x.copy()
            xx[i] = xx[i] + self.g_delta
            hessian[:,i] = (g(xx) - gx) / self.g_delta
        return hessian
    
    
    def line_search_exact(self, x_k, s_k):
    # exact line search method, gives alphak
        
        def step_function(alpha, x_k, s_k):
            return self.function(x_k + alpha*s_k)
        
        guess = self.alpha_k # Guess for the scipy optimizer. Don't know what is a reasonable guess. Maybe alpha_k-1
        
        self.alpha_k = optimize.minimize(step_function, guess, args=(x_k,s_k)) 
        #^above updates the self.alpha_k to be the new one 
        #below returns the new alpha_k. Don't know what is better
        return alpha_k
    
    def line_search_inexact(self, x_k, s_k):
        # inexact line search method, gives alphak
        def step_function(alpha, x_k, s_k):
            return self.function(x_k + alpha*s_k)
        
        #recommended default values



    