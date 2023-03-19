# Import packages
import numpy as np
from IPython.display import clear_output


### CREATE RPCA CLASS ###
class RPCA:

# Perform robust principal component analysis (RCPA) using the alternating directions method (ADM)
# Code from Brunton & Kutz (2022), https://doi.org/10.1017/9781009089517

# Initiation and fit are done identically to sk-learn's PCA class

    def __init__(self): # Initialise class
        pass

    # Define the shrink function
    def shrink(self, X, tau):
        return np.sign(X) * np.maximum((np.abs(X) - tau), np.zeros_like(X))

    # Define the SVT function
    def SVT(self, X, tau):
        U, S, V = np.linalg.svd(X, full_matrices = False) # Compute the economy SVD
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    # Perform the PCP iteration
    def fit(self, X, tol = None, imax = 1000, verbose = (False, 0), mu = None, lmbda = None, iter_output = False, save_best = False):
        if verbose :
            print('Fit initiated ... ')

        # Initiate iteration parameters
        if mu :
            self.mu = mu
        else : # If mu is not specified
            self.mu = np.prod(X.shape) / (4 * np.sum(np.abs(X.reshape(-1))))
        self.mu_inv = 1 / self.mu

        if lmbda :
            self.lmbda = lmbda
        else : # If lambda is not specified
            self.lmbda = 1 / np.sqrt(np.max(X.shape))
        iter = 0
        err = np.Inf; err_min = np.Inf
        if tol :
            _tol = tol
        else : # If tolerance is not specified
            _tol = 1E-7 * np.linalg.norm(X, ord = 'fro')

        # Initiate matrices S, Y & L
        S = np.zeros_like(X) 
        Y = np.zeros_like(X)
        L = np.zeros_like(X)

        # Enter PCP loop
        while (err > _tol) and (iter < imax) :

            # Update L :    L_(k+1) = SVT(X - S_(k) - Y_(k))
            L = self.SVT(X - S + self.mu_inv * Y, self.mu_inv)     
            # Update S :    S_(k+1) = shrink(X - L_(k+1) + 1/mu * Y_(k))
            S = self.shrink(X - L + (self.mu_inv * Y), self.mu_inv * self.lmbda)   
            # Update Y :    Y_(k+1) = Y_(k) + 1/mu * (X - L_(k+1) - S_(k+1)) 
            Y = Y + self.mu * (X - L - S)   
            
            # Update the error
            err = np.linalg.norm(X - L - S, ord = 'fro')
            iter += 1

            if save_best and err < err_min :
                Lb = L.copy()
                Sb = S.copy()
                err_min = err

            if verbose[0] and iter % verbose[1] == 0 :
                clear_output(wait = True)
                prog = _tol/err
                bar = '|' + round(20*prog)*'/' + (20 - round(20*prog)) * '_' + '|'
                print(f'... {round(prog, 3)} ---> 1.0 ... {bar} ... ({iter}/{imax}) ...')

            # Print number of iterations required
        if verbose[0] :
            clear_output()
            if (iter == imax) and (err > _tol):
                print(f'No convergence after {iter} iterations')
            else:
                print(f'Solution found after {iter} iterations')

        if save_best :
            if iter_output :
                return (L, S), (Lb, Sb), iter
            else :
                return (L, S), (Lb, Sb)
        else :
            if iter_output :
                return L, S, iter
            else :
                return L, S