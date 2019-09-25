'''
    Script containing an optimization problem class, which contains methods for
    solving optimization problems using different methods
                                                                            '''
import numpy as np
import scipy

#todo: beräkna G och sedan H^0 = G^-1
class Problem(object):
    def __init__(self, objective_function, dimensions, gradient=None):
        self.objective_function = objective_function
        self.dimensions = dimensions
        if gradient != None:
            self.gradient = gradient
    

class Solver(object):
    def __init__(self, problem, g_tolerance=10^(-6), g_delta=10^(-8)):
        self.objective_function = problem.objective_function
        self.dimensions = problem.dimensions
        if gradient != None:
            self.gradient = problem.gradient
        self.g_tolerance = g_tolerance
        self.g_delta = g_delta
    
    @methodclass
    def optimize(cls,opt_object,methods):
        
    def compute_gradient(self, x_k):
        # Does the explicit gradient function exist? Then use it!
        if self.gradient != None:
            return self.gradient(x_k)
        
        # If not, then we compute it numerically
        n = self.dimensions
        gradient_k = np.zeros(1,n)
        x = np.zeros(n,n)

        for i in range(n):
            x = x_k.copy()
            x[i] = x[i] + delta
            gradient_k[i] = (self.objective_function(x)-self.objective_function(x_k))/self.g_delta

        return gradient_k


    def LineSearchExact(self, x_k, s_k):
    # exact line search method, gives alphak
        
        def step_function(alpha,x_k,s_k):
            return self.function(x_k + alpha * s_k)
        
        guess = self.alpha_k # Guess for the scipy optimizer. Don't know what is a reasonable guess. Maybe alpha_k-1
        
        self.alpha_k = optimize.minimize(step_function, guess, args=(x_k,s_k)) 
        #^above updates the self.alpha_k to be the new one 
        #below returns the new alpha_k. Don't know what is better
        return alpha_k



    