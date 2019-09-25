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
        x = np.zeros(1,n)
        for i in range(n):
            delta_x[i] = x_k[i]-x_km1[0]
            x[i] = 1 #Concatenate x[:i-1] delta_x[i] x[i+1:]  
            gradient_k[i] = self.objective_function(x[i])/delta_x[i]
        return gradient_k





    