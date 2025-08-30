import numpy as np
from numpy.typing import NDArray

# Accept input as numpy array

class GaussianMixtureModel():
    def __init__(self, n_components: int, n_iter: int, reg_covar= 1e-6, tol=1e-3, verbose=False):
        self.n_components = n_components
        self.n_iter = n_iter
        self.reg_covar = reg_covar
        self.tol = tol
        self.verbose = verbose
        self.pis_ = None # [K,]
        self.mus_ = None # [K, D]
        self.covs_ = None # [K, D, D]

    def multivariate_gaussian(self, x, mu, cov):
        N, D = x.shape
        diff = x - mu # [N, D]
        det_cov = np.linalg.det(cov) # []
        inv_cov = np.linalg.inv(cov) # [D, D]
        
        # Normalization constant
        norm_const = 1 / np.sqrt((2 * np.pi) ** D * det_cov) # []

        # Exponential term
        exp_term = np.exp(-0.5 * np.einsum('ni, ij, nj -> n', diff, inv_cov, diff)) # [N,]

        return norm_const * exp_term # [N,]

    def fit(self, X: NDArray[np.float64]):
        N, D = X.shape
        K = self.n_components

        # Initialization 
        pis = np.ones(K) / K # [K,]
        mus = X[np.random.choice(N, K, replace=False)] # [K, D]
        covs = np.array([np.eye(D) for _ in range(K)]) # [K, D, D]

        prev_ll = None

        for i in range(self.n_iter):
            # E-step
            resp = np.zeros([N, K]) # [N, K]
            for k in range(K):
                resp[:, k] = pis[k] * self.multivariate_gaussian(X, mus[k], covs[k])
            resp_sum = resp.sum(axis=1, keepdims=True) # [N, 1]
            resp /= resp_sum # [N, K]

            # M-step
            Nk = resp.sum(axis=0) # [K,]
            pis = Nk / N # [K,]
            mus = resp.T @ X / Nk[:, np.newaxis] # [K, D]
            for k in range(K):
                covs[k] = (resp[:, k][:, np.newaxis] * (X - mus[k])).T @ (X- mus[k]) / Nk[k]
                covs[k] += self.reg_covar * np.eye(D)
            
            # Log-likelihood
            ll = np.sum(np.log(resp_sum))
            if self.verbose:
                print(f"Iter {i} log-likelihood: {ll:.3f}")
            if prev_ll is not None and abs(prev_ll - ll) < self.tol:
                break
            prev_ll = ll
        self.pis_ = pis
        self.mus_ = mus
        self.covs_ = covs
        return self
    
    def predict_proba(self, X):
        N, D = X.shape
        K = self.n_components

        resp = np.zeros([N, K]) # [N, K]
        for k in range(K):
            resp[:, k] = self.pis_[k] * self.multivariate_gaussian(X, self.mus_[k], self.covs_[k])
        resp_sum = resp.sum(axis=1, keepdims=True) # [N, 1]
        return resp / resp_sum # [N, K]
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

