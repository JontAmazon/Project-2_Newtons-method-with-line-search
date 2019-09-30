"""
    Script containing an optimization problem class, which contains methods for
    solving optimization problems using different methods
"""


import numpy as np
import scipy.linalg as sl
import scipy


class Problem(object):
    def __init__(self, objective_function, gradient_function=None, hessian_function=None):
        self.objective_function = objective_function
        self.gradient_function = gradient_function
        self.hessian_function = hessian_function
        

class Solver(object):
    def __init__(self, problem, dimensions=None, tol=1e-5, max_iterations=400, grad_tol=1e-6, hess_tol=1e-3):
        self.objective_function = problem.objective_function
        self.gradient_function = problem.gradient_function #(might be equal to None).
        self.hessian_function = problem.hessian_function #(might be equal to None).
        self.tol = tol
        self.grad_tol = grad_tol
        self.hess_tol = hess_tol
        self.max_iterations = max_iterations
        if dimensions is not None:
            self.dimensions = dimensions #this can probably be removed in the final version.
                                        #just want it for debugging purposes.


    def find_local_min(self, quasi_newton_method, x0, line_search_method=None, debug=False):
        """Solves the problem of finding a local minimum of the function 
            described in the input problem, using a Quasi-Newton method
            together with line search.
        """
        self.debug = debug
        self.dimensions = len(x0)

        if debug:                           # TODO: ersätt med:  if rosenbrock
            def gradient(x): #EV TODO: ersätt med: "exact_gradient()"
                grad = np.zeros((2,1))
                grad[0,0] = 400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2
                grad[1,0] = -200*x[0]**2 + 200*x[1]
                #print('grad '+ str(grad))
                return grad
            
            def G(x):        #EV TODO: ersätt med "exact_hessian()"
                hess = np.zeros((2,2))
                hess[0,0] = 1200*x[0]**2 - 400*x[1] + 2
                hess[0,1] = hess[1,0] = -400*x[0]
                hess[1,1] = 200
                return hess
            
        #Create an empty vector for the encountered x-values during minimization  
        x_values = [];
        
        #[Task 12]. Also create two vectors for studying H by
        #comparing it to H_correct, where H_correct = inv(G(x)).
        h_diff_values = [];
        h_quotient_values = [];
        
        x0 = np.array(x0).astype(float)
        x_km1 = x0*0
        x_k = x0.reshape(self.dimensions,1) # Reshape the x_k to fit with the gradients and stuff
        x_kp1 = x0
        
        g = self.compute_gradient(x_k)
        H = sl.inv(self.compute_hessian(x_k))

        for i in range(self.max_iterations): 
            
            #Save the current x_k in a list for plotting
            x_values.append(x_k)
            if self.dimensions==2: # TASK 12.
                h_diff_values.append(sl.norm(H - sl.inv(G(x_k)),2))
                h_quotient_values.append(sl.norm(H,2) / sl.norm(sl.inv(G(x_k),2)))
            
            if self.debug:
                print('                 #' + str(i))
                print('x_k:             ' + str(x_k.T[0]))
                print('f(x_k):          ' + str(self.objective_function(x_k)))                
                #print('||g - g_corr||:  ' + str(sl.norm(g - gradient(x_k),2)))
                #print('||H - H_corr||:  ' + str(sl.norm(H - sl.inv(G(x_k)),2)))

            if sl.norm(g, 2) < self.tol:
                #print('||g|| < tol ==> We are done if only G > 0')
                hess = self.compute_hessian(x_kp1)
                if self.is_positive_definite(hess):
                    print('Yaaay! Local minima found after ' + str(i) + ' iterations.')
                    print('Optimal x: ' + str(x_k.T))
                    print('Optimal f: ' + str(self.objective_function(x_k)))
                    return x_k, self.objective_function(x_k), x_values, h_diff_values, h_quotient_values
                #print('Sadly, it was not the case.\n')
            s_k = -(H @ g) #Newton direction
            alpha = self.line_search(line_search_method, x_k, s_k)
            if self.debug:
                print('alpha: ' + str(alpha))
                print()
            
            x_kp1 = x_k + alpha*s_k
            g = self.compute_gradient(x_kp1)
            H = self.quasi_newton(quasi_newton_method, H, x_k, x_km1)
            x_km1 = x_k
            x_k=x_kp1
        
        print('Local minima could not be found in ' \
            + str(self.max_iterations) + ' iterations.')
        return x_k, self.objective_function(x_k), x_values, h_diff_values, h_quotient_values
    
    # Methods to compute the inverse Hessian. All are accessed through the quasi_newton method below.
    def exact_newton(self, H, x_k, x_km1):
        #Of the 3 in-parameters, we only use x_k.
        return sl.inv(self.compute_hessian(x_k))
        
    def good_broyden(self, H, x_k, x_km1):
        # Superposition of np column "matrix" results in concatenation,
        # so we need to transpose delta_k to row matrix and then transpose back
        delta_k = (x_k.T - x_km1.T).T
        gamma_k = self.compute_gradient(x_k)-self.compute_gradient(x_km1)
        # u and a are just temporary variables used to
        # increase the readability of the return statement, 
        # which are defined as in the slides
        u = (delta_k.T - (H @ gamma_k).T).T
        a = float(1/(u.T@gamma_k))
        return H + a*u@u.T

    def bad_broyden(self, H,x_k,x_km1):
        # Superposition of np column "matrix" results in concatenation,
        # so we need to transpose delta_k to row matrix and then transpose back
        delta_k = (x_k.T - x_km1.T).T
        gamma_k = self.compute_gradient(x_k)-self.compute_gradient(x_km1)
        u = delta_k - H @ gamma_k
        a = float(1/(gamma_k.T@gamma_k))
        return H + a*u@gamma_k.T

    def davidon_fletcher_powell(self,H,x_k,x_km1):
        # Superposition of np column "matrix" results in concatenation,
        # so we need to transpose delta_k to row matrix and then transpose back
        delta_k = (x_k.T - x_km1.T).T
        gamma_k = self.compute_gradient(x_k)-self.compute_gradient(x_km1)
        a = (delta_k@delta_k.T)/float(delta_k.T@gamma_k)
        b = H@np.outer(gamma_k,gamma_k)@H/(gamma_k.T@H@gamma_k)

        return H + a - b
    
    def broyden_fletcher_goldfarb_shanno(self,H,x_k,x_km1):
        # Superposition of np column "matrix" results in concatenation,
        # so we need to transpose delta_k to row matrix and then transpose back
        delta_k = (x_k.T - x_km1.T).T
        gamma_k = self.compute_gradient(x_k) - self.compute_gradient(x_km1)
        inner = float(gamma_k.T@H@gamma_k/(float(delta_k.T@gamma_k)))
        outer = (delta_k@delta_k.T)/float(delta_k.T@gamma_k)
        a = (1+inner)*outer
        b = (delta_k@gamma_k.T@H + H@gamma_k@delta_k.T)/(float(delta_k.T@gamma_k))
        return H + a - b

    # Python-switch statement that calls the relevant quasi newton method.
    def quasi_newton(self, quasi_newton_method, H, x_k, x_km1):
        method = {'exact_newton' : self.exact_newton,
            'good_broyden' : self.good_broyden,
            'bad_broyden' : self.bad_broyden,
            'davidon_fletcher_powell' : self.davidon_fletcher_powell,
            'broyden_fletcher_goldfarb_shanno' : self.broyden_fletcher_goldfarb_shanno,
        }
        return method[quasi_newton_method](H, x_k, x_km1)
        raise Exception('Invalid input for Quasi-Newton method.')
    
    
    
    def line_search(self, line_search_method, x_k, s_k):
       """
           Returns alpha by the chosen line search method.
       """
       if line_search_method==None:
           return 1
       if line_search_method=='exact_line_search':
           return self.exact_line_search(x_k, s_k)
       if line_search_method=='wolfe-powell':
           return self.inexact_line_search('wolfe-powell', x_k, s_k)
       if line_search_method=='goldstein':
           return self.inexact_line_search('goldstein', x_k, s_k)
       raise Exception('Invalid input for line search method.')
    
    def exact_line_search(self, x_k, s_k):
    # exact line search method, gives alphak
        
        def step_function(alpha, x_k, s_k):
            return self.objective_function(x_k + alpha*s_k)
        x_copy = x_k.copy().reshape(self.dimensions,1)
        guess = 1 # Guess for the scipy optimizer. Don't know what is a reasonable guess. Maybe alpha_k-1. Or just 1?
        optimization_res = scipy.optimize.minimize(step_function, guess, args=(x_copy,s_k)) #returns some kind of optimization object, so we need to extract the x-value
        alpha_k = optimization_res.x
        #below returns the new alpha_k. Don't know what is better
        return alpha_k
    
    
    def inexact_line_search(self, line_search_method, x_k, s_k):
        """
            Inexact line search method for computing alpha^(k), using either 
            Wolfe-Powell or Goldstein conditions.
        """
        #Define the default values for the method parameters
        self.rho = 0.1
        self.sigma = 0.7
        self.tao = 0.1
        self.chi = 9
            
        #Initiate alpha_L and alpha_U
        alpha_L = 0
        alpha_U = 10**2
        
        #Initiate alpha_0 by taking the average of the boundary values
        #alpha_0 = (alpha_L + alpha_U)/2
        alpha_0 = 1
        #ALTERNATIVELY: alpha_0 = np.random.rand(alpha_L, alpha_U, 1)
            
        #Compute the initial values of the function and the corresponding gradients
        f_alpha_0, f_alpha_L, df_alpha_0, df_alpha_L = self.compute_f_and_df(alpha_0, alpha_L,x_k,s_k)
            
        #Initiate the boolean values of lc and rc 
        lc = False
        rc = False
         #Check if the conditions are fullfilled, return booleans lc and rc
        if line_search_method=='wolfe-powell':
            lc, rc = self.lc_rc_wolfe_powell(alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                                        f_alpha_L, df_alpha_0, df_alpha_L)
        if line_search_method=='goldstein':
            lc, rc = self.lc_rc_goldstein(alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                                        f_alpha_L, df_alpha_0, df_alpha_L)
            
        while (not lc or not rc):
            if not lc:
                #Implementation of Block 1 in the slides
                #if abs(df_alpha_L - df_alpha_0) <= 0.0001:
                #    print("Look here")
                delta_alpha_0 = (alpha_0 - alpha_L)*df_alpha_0/(df_alpha_L - df_alpha_0) #Compute delta(alpha_0) by extrapolation
                delta_alpha_0 = np.max([abs(delta_alpha_0), self.tao*(alpha_0 - alpha_L)]) #Make sure delta_alpha_0 is not too small                
                delta_alpha_0 = np.min([delta_alpha_0, self.chi*(alpha_0 - alpha_L)]) #Make sure delta_alpha_0 is not too large
                alpha_L = np.copy(alpha_0) #Assign the value of alpha_0 to alpha_L
                alpha_0 = alpha_0 + delta_alpha_0#Update the value of alpha_0
            else:
                #Implementation of Block 2 in the slides
                alpha_U = np.min([alpha_0, alpha_U]) #Update the lower bound
                bar_alpha_0 = ((alpha_0 - alpha_L)**2)*df_alpha_L/2*(f_alpha_L - f_alpha_0 + (alpha_0 - alpha_L)*df_alpha_L) #Compute bar(alpha_0) by interpolation
                bar_alpha_0 = np.max([bar_alpha_0, alpha_L + self.tao*(alpha_U - alpha_L)]) #Make sure bar_alpha_0 is not too small
                bar_alpha_0 = np.min([bar_alpha_0, alpha_U - self.tao*(alpha_U - alpha_L)]) #Make sure bar_alpha_0 is not too large
                alpha_0 = bar_alpha_0 # Update the value of alpha_0
                
            #Compute the function values and their corresponing gradients
            f_alpha_0, f_alpha_L, df_alpha_0, df_alpha_L = self.compute_f_and_df(alpha_0, alpha_L, x_k, s_k)
            
            #Return the boolean values of lc and rc for the next iteration
            if line_search_method=='wolfe-powell':
                lc, rc = self.lc_rc_wolfe_powell(alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                                        f_alpha_L, df_alpha_0, df_alpha_L)
            if line_search_method=='goldstein':
                lc, rc = self.lc_rc_goldstein(alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                                        f_alpha_L, df_alpha_0, df_alpha_L)
            
        return alpha_0#, f_alpha_0  # This gave an increasing dimensions of alpha...if we don't need the f_alpha_0 keep this commented

    def compute_f_and_df(self, alpha_0, alpha_L, x_k, s_k):
        '''Computes the function and the corresponding gradient evaluated 
        at alpha_0.
                                                                        '''
        x_copy = x_k.copy().reshape(self.dimensions,1)
        #Define the values on which to evaluate the function and the gradient
        alpha_0_eval = x_copy + alpha_0 * s_k
        alpha_L_eval = x_copy + alpha_L * s_k
        
        #Evaluate the gradient for the two points defined above (using the chain rule, thus s_k)
        g0l = self.compute_gradient(alpha_0_eval)
        df_alpha_0 = float(self.compute_gradient(alpha_0_eval).T @ s_k)
        df_alpha_L = float(self.compute_gradient(alpha_L_eval).T @ s_k)
        #Evaluate the function in the same points
        f_alpha_0 = float(self.objective_function(alpha_0_eval))
        f_alpha_L = float(self.objective_function(alpha_L_eval))
        
        return f_alpha_0, f_alpha_L, df_alpha_0, df_alpha_L
    

    def lc_rc_wolfe_powell(self, alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                           f_alpha_L, df_alpha_0, df_alpha_L):
        '''
        Returns lc = True and rc = True if the Wolfe-Powell conditions
        are fulfilled for alpha_0 and alpha_L.
                                                                        '''                
        #Define the boolean return variables
        lc = False
        rc = False
            
        if df_alpha_0 >= self.sigma * df_alpha_L:                      # this is regular Wolfe
            lc = True

#        if abs(df_alpha_0) <= self.sigma * abs(df_alpha_L):           # this is strong Wolfe
#            lc = True
    
        if f_alpha_0 <= f_alpha_L + self.rho*(alpha_0 - alpha_L)*df_alpha_L:
            rc = True
            
        #TODO: Should we use the strong Wolfe condition as well?
                
        return lc, rc

    def lc_rc_goldstein(self, alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                           f_alpha_L, df_alpha_0, df_alpha_L):
        '''TODO'''
        #Define the boolean return variables
        lc = False
        rc = False
        
        if f_alpha_0 >= f_alpha_L + (1-self.rho)*(alpha_0-alpha_L)*df_alpha_L:
            lc = True
            
        if f_alpha_0 <= f_alpha_L + self.rho*(alpha_0-alpha_L)*df_alpha_L:
            rc = True
            
        return lc, rc
        


    def compute_gradient(self, x):
        # Do we have an explicit function for the gradient? Then use it!
        if self.gradient_function != None:
            g = self.gradient_function(x)
            g = np.array([g[i] for i in range(len(g))])
            return g
        
        # If not, compute it with central finite differences:
        n = self.dimensions
        gradient = np.zeros((n,1))
        f = self.objective_function
        delta = self.grad_tol
        #delta = self.grad_tol * sl.norm(x, 2)
        
        for i in range(n):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] = x1[i] + delta
            x2[i] = x2[i] - delta
            gradient[i][0] = (f(x1) - f(x2)) / (2*delta)
        return gradient
    
    def compute_hessian(self, x):
        # The i:th column of the Hessian G_i equals g(x) differentiated w.r.t. x_i
        # This is approximated with a finite difference:
        # G_i = (g(x + tol*e_i) - g(x))/tol
        if self.hessian_function != None:
            return self.hessian_function(x)

        n = self.dimensions
        hessian = np.zeros((n,n))
        g = self.compute_gradient
        delta = self.hess_tol
        #delta = self.hess_tol * sl.norm(x, 2)
        
        for i in range(n):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] = x1[i] + delta
            x2[i] = x2[i] - delta            
            hessian[:,i] = ((g(x1) - g(x2)) / (2*delta)).T
        hessian = 1/2*hessian + 1/2*np.conj(hessian.T)
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

    



    ''' CODING INEXACT LINE SEARCH AGAIN... TO MAYBE DISCOVER AN ERROR '''
    def inexact_line_search2(self, line_search_method, x_k, s_k):
        """
            Inexact line search method for computing alpha^(k), using either 
            Wolfe-Powell or Goldstein conditions.
        """
        self.rho = 0.1
        self.sigma = 0.7
        self.tao = 0.1
        self.chi = 9
            
        a_L = 1e-3
        a_U = 1e5    
        a_0 = 1 # testa 1e1
            
        f_alpha_0, f_alpha_L, df_alpha_0, df_alpha_L = self.compute_f_and_df(alpha_0, alpha_L,x_k,s_k)
            
        if line_search_method=='wolfe-powell':
            lc, rc = self.lc_rc_wolfe_powell(alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                                        f_alpha_L, df_alpha_0, df_alpha_L)
        if line_search_method=='goldstein':
            lc, rc = self.lc_rc_goldstein(alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                                        f_alpha_L, df_alpha_0, df_alpha_L)
        
        
    #delta_alpha_0
    #df_alpha_0
    #df_alpha_L
    #f_alpha_0
    #f_alpha_L
    #bar_alpha_0
        while (not lc or not rc):
            if not lc:
                pass
                #Block 1:
                #...
                
            else:
                pass
                #Block 2:
                #...
                
            f_alpha_0, f_alpha_L, df_alpha_0, df_alpha_L = self.compute_f_and_df(alpha_0, alpha_L, x_k, s_k)
            
            if line_search_method=='wolfe-powell':
                lc, rc = self.lc_rc_wolfe_powell(alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                                        f_alpha_L, df_alpha_0, df_alpha_L)
            if line_search_method=='goldstein':
                lc, rc = self.lc_rc_goldstein(alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                                        f_alpha_L, df_alpha_0, df_alpha_L)
            
        return a_0












        