'''
    Script containing an optimization problem class, which contains methods for
    solving optimization problems using different methods
                                                                            '''
import numpy as np
import scipy.linalg as sl

''' QUESTIONS '''
#   - What do we do if G > 0 is not the case when we reach g=0?


class Problem(object):
    def __init__(self, objective_function, dimensions, gradient=None):
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.gradient = gradient #(might be equal to None).
        

class Solver(object):
    def __init__(self, problem, tol=1e-6, max_iterations=100):
        self.objective_function = problem.objective_function
        self.dimensions = problem.dimensions
        self.gradient_function = problem.gradient #(might be equal to None).
        self.tol = tol
        self.max_iterations = max_iterations
        #todo: beräkna G och H0 = G^-1


    def find_local_min(self, quasi_newton_method, line_search_method, x0):
        """Solves the problem of finding a local minimum of the function 
            described in the input problem, using a Quasi-Newton method
            together with line search.
        """
        x = x0
        g = self.compute_gradient(x)
        H = sl.inv(self.compute_hessian(x))
        for i in range(self.max_iterations):
            s = -H @ g #Newton direction
            alpha = self.line_search(line_search_method) # plus fler inparametrar
            x = x + alpha*s
            
            g = self.compute_gradient(x)
            if sl.norm(g, 2) < self.tol:
                G = self.compute_hessian(x)
                if self.is_positive_definite(G):
                    print('Local minima found!')
                    return x
            H = self.quasi_newton(quasi_newton_method, H) # plus fler inparametrar?
        
        else:
            print('Local minima could not be found in ' \
                  + str(self.max_iterations) + ' iterations.')
    
    def line_search(self, line_search_method): # plus fler inparametrar
        '''TODO'''
        # switch-sats
        pass
    
    def quasi_newton(self, quasi_newton_method, H): # plus fler inparametrar?
        '''TODO'''
        # switch-sats
        pass
        
    
    
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
        delta = 1e-4
        for i in range(n):
            xx = x.copy()
            xx[i] = xx[i] + delta
            gradient[i] = (f(xx) - fx) / delta
        return gradient
    
    def compute_hessian(self, x):
        # The i:th column of the Hessian G_i equals g(x) differentiated w.r.t. x_i
        # This is approximated with a finite difference:
        # G_i = (g(x + tol*e_i) - g(x))/tol
        n = self.dimensions
        hessian = np.zeros((n,n))
        g = self.compute_gradient
        gx = g(x) #we only need to calculate this once
        delta = 1e-4
        for i in range(n):
            xx = x.copy()
            xx[i] = xx[i] + delta
            hessian[:,i] = (g(xx) - gx) / delta
        return hessian

    def is_positive_definite(self, A):                                          #[92%]
        # Computing the Cholesky decomposition with (numpy.linalg.cholesky)
        # raises LinAlgError if the matrix is not positive definite.
        try:
            sl.cholesky(A)
        except sl.LinAlgError:
            return False
        else:
            return True
    