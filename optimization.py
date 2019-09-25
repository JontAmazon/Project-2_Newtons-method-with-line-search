'''
    Script containing an optimization problem class, which contains methods for
    solving optimization problems using different methods
                                                                            '''
import numpy as np
import scipy

class Optimization(object):

    def __init__(self,objective_function,gradient=None):
        self.objective_function = objective_function
        if gradient != None:
            self.gradient = gradient
    
    @methodclass
    def optimize(cls,opt_object,methods):
        
    def compute_gradient(self,x_k,x_km1):
        if self.gradient != None:
            return self.gradient(x_k)
        n = len(x_k)
        gradient_k = np.zeros(1,len(x_k))
        delta_x = np.zeros(1,n)
        x = np.zeros(n,n)
        
        delta_x[0] = x_k[0]-x_km1[0]
        x[:][0] = np.hstack((delta_x[0],x[1:]))
        gradient_k[0] = self.objective_function(x[:][0])/delta_x[0]

        for i in range(1,n-1):
            delta_x[i] = x_k[i]-x_km1[i]
            x[:][i] = np.hstack((x[:i-1],delta_x[i],x[i+1]))
            gradient_k[i] = self.objective_function(x[:][i])/delta_x[i]

        delta_x[n] = x_k[n]-x_km1[n]
        x[:][0] = np.hstack((x[:-1], delta_x[n]))
        gradient_k[n] = self.objective_function(x[:][n])/delta_x[n]
        
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



    