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
        x_km1 = x0*0.9
        x_k = x0
        x_kp1 = x0
        
        g = self.compute_gradient(x)
        H = sl.inv(self.compute_hessian(x))
        
        for i in range(self.max_iterations):
            s_k = -H @ g #Newton direction
            alpha = self.line_search(line_search_method) # plus fler inparametrar
            x_kp1 = x_k + alpha*s_k
            
            g = self.compute_gradient(x_kp1)
            if sl.norm(g, 2) < self.tol:
                G = self.compute_hessian(x_kp1)
                if self.is_positive_definite(G):
                    print('Local minima found!')
                    return x
            H = self.quasi_newton(quasi_newton_method, H, x_k, x_km1) # plus fler inparametrar?
            x_km1 = x_k
            x_k=x_kp1
        
        print('Local minima could not be found in ' \
            + str(self.max_iterations) + ' iterations.')
    
    
    # Methods to compute the inverse Hessian. All are accessed through the quasi_newton method below.
    def good_broyden(self,H,x_k,x_km1):
        delta_k = x_k - x_km1
        gamma_k = self.compute_gradient(x_k)-self.compute_gradient(x_km1)
        # u and a are just temporary variables used to
        # increase the readability of the return statement
        u = delta_k - H @ gamma_k
        a = np.divide(1,u.T@gamma_k)
        return H + a@u@u.T

    def bad_broyden(self, H,x_k,x_km1):
        return sl.inv(self.compute_hessian(x_k))

    def davidson_fletcher_powell(self,H,x_k,x_km1):
        delta_k = x_k - x_km1
        gamma_k = self.compute_gradient(x_k)-self.compute_gradient(x_km1)
        return H + sl.inv(delta_k.T@gamma_k)@(delta_k@delta_k.T) - \
            sl.inv(gamma_k.T@H@gamma_k)@(H@gamma_k@gamma_k.T@H)
    
    def broyden_fletcher_goldfarb_shanno(self,H,x_k,x_km1):
        delta_k = x_k - x_km1 
        gamma_k = self.compute_gradient(x_k) - self.compute_gradient(x_km1)
        a = (1 + sl.inv(delta_k.T@gamma_k)@(gamma_k.T@H@gamma_k))@\
            (sl.inv(delta_k.T@gamma_k)@delta_k@delta_k.T)
        b = sl.inv(delta_k.T@gamma_k)@(delta_k@gamma_k.T@H + H@gamma_k@delta_k.T)
        return H + a - b

    # Python-switch statement that calls the relevant quasi newton method.
    def quasi_newton(self, quasi_newton_method, H, x_k, x_km1): # plus fler inparametrar?
        method = {'good_broyden' : good_broyden,
            'bad_broyden' : bad_broyden,
            'davidson_fletcher_powell' : davidson_fletcher_powell,
            'broyden_fletcher_goldfarb_shanno' : broyden_fletcher_goldfarb_shanno,
        }
        return method[quasi_newton_method](self,H,x_k,x_km1)
    
    def line_search_exact(self, x_k, s_k):
    # exact line search method, gives alphak
        
        def step_function(alpha, x_k, s_k):
            return self.objective_function(x_k + alpha*s_k)
        
        guess = self.alpha_k # Guess for the scipy optimizer. Don't know what is a reasonable guess. Maybe alpha_k-1
        
        self.alpha_k = optimize.minimize(step_function, guess, args=(x_k,s_k)) 
        #^above updates the self.alpha_k to be the new one 
        #below returns the new alpha_k. Don't know what is better
        return alpha_k
    
    def line_search_inexact(self, x_k, s_k):
        # Inexact line search method for computing alpha^(k)
        
        def lc_rc_wolfe_powell(self, alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                               f_alpha_L, df_alpha_0, df_alpha_L):
            '''
            Returns lc = True and rc = True if the Wolfe-Powell conditions
            are fulfilled for alpha_0 and alpha_L.
                                                                            '''                
            #Define the boolean return variables
            lc = False
            rc = False
                
            if df_alpha_0 >= self.sigma * df_alpha_L:
                lc = True
        
            if f_alpha_0 <= f_alpha_L + self.rho*(alpha_0 - alpha_L)*df_alpha_L:
                rc = True
                
            #TODO: Should we use the strong Wolfe condition as well?
                    
            return lc, rc
        
        
        def compute_f_and_df(self, alpha_0, alpha_L):
            '''Computes the function and the corresponding gradient evaluated 
            at alpha_0.
                                                                            '''
            #Define the values on which to evaluate the function and the gradient
            alpha_0_eval = x_k + alpha_0 * s_k
            alpha_L_eval = x_k + alpha_L * s_k
            
            #Evaluate the gradient for the two points defined above (using the chain rule, thus s_k)
            df_alpha_0 = self.compute_gradient(alpha_0_eval).T * s_k
            df_alpha_L = self.compute_gradient(alpha_L_eval).T * s_k
            
            #Evaluate the function in the same points
            f_alpha_0 = self.objective_function(alpha_0_eval)
            f_alpha_L = self.objective_function(alpha_L_eval)
            
            return f_alpha_0, f_alpha_L, df_alpha_0, df_alpha_L
        
            
        #Define the default values for the method parameters
        self.rho = 0.1
        self.sigma = 0.7
        self.tao = 0.1
        self.chi = 9
            
        #Initiate alpha_L and alpha_U
        alpha_L = 0
        alpha_U = 10**99
        
        #Initiate alpha_0 by taking the average of the boundary values
        alpha_0 = (alpha_L + alpha_U)/2
        #ALTERNATIVELY: alpha_0 = np.random.rand(alpha_L, alpha_U, 1)
            
        #Compute the initial values of the function and the corresponding gradients
        f_alpha_0, f_alpha_L, df_alpha_0, df_alpha_L = compute_f_and_df(alpha_0, alpha_L)
            
        #Initiate the boolean values of lc and rc 
        lc = False
        rc = False
            
        while (not lc and not rc):
                
            if not lc:
                #Implementation of Block 1 in the slides
                    
                delta_alpha_0 = (alpha_0, alpha_L)*df_alpha_0/(df_alpha_L - df_alpha_0) #Compute delta(alpha_0) by extrapolation
                delta_alpha_0 = np.max(delta_alpha_0, self.tao*(alpha_L - alpha_L)) #Make sure delta_alpha_0 is not too small
                delta_alpha_0 = np.min(delta_alpha_0, self.chi*(alpha_L - alpha_L)) #Make sure delta_alpha_0 is not too large
                alpha_L = np.copy(alpha_0) #Assign the value of alpha_0 to alpha_L
                alpha_0 = alpha_0 + delta_alpha_0#Update the value of alpha_0
            else:
                #Implementation of Block 2 in the slides
                    
                alpha_U = np.min(alpha_0, alpha_U)
                bar_alpha_0 = ((alpha_0 - alpha_L)**2)*df_alpha_L/2*(f_alpha_L - f_alpha_0 + (alpha_0 - alpha_L)*df_alpha_L) #Compute bar(alpha_0) by interpolation
                bar_alpha_0 = np.max(bar_alpha_0, alpha_L + self.tao*(alpha_L - alpha_L)) #Make sure bar_alpha_0 is not too small
                bar_alpha_0 = np.min(bar_alpha_0, alpha_U - self.tao*(alpha_L - alpha_L)) #Make sure bar_alpha_0 is not too large
                alpha_0 = bar_alpha_0 #Update the value of alpha_0
                
            f_alpha_0, f_alpha_L, df_alpha_0, df_alpha_L = compute_f_and_df(alpha_0, alpha_L)
            lc, rc = lc_rc_wolfe_powell(alpha_0, alpha_L, x_k, s_k, f_alpha_0, \
                                    f_alpha_L, df_alpha_0, df_alpha_L)
        
        return alpha_0, f_alpha_0
    
#    def line_search(self, line_search_method, x_k, s_k): # plus fler inparametrar
#            method = {'line_search_inexact' : line_search_inexact,
#            'line_search_exact' : line_search_exact,
#            }
#            return method[line_search_method](self,x_k,s_k)

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
            x_copy = x.copy()
            x_copy[i] = x_copy[i] + delta
            gradient[i] = (f(x_copy) - fx) / delta
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
    