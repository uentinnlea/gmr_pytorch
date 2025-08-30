import numpy as np
from .gmm import GaussianMixtureModel

class GaussianMixtureRegression(GaussianMixtureModel):
    def __init__(self, n_components, n_iter, reg_covar= 1e-6, tol=1e-3, verbose=False):
        super().__init__(n_components, n_iter, reg_covar, tol, verbose)

    def full_mixture(self, input_data, input_idx, output_idx):
        pi, mu, cov = self.pis_, self.mus_, self.covs_
        N = input_data.shape[0]
        K = pi.shape[0]
        n_out = len(output_idx)

        all_means = np.zeros([K, N, n_out])
        all_covs = np.zeros([K, n_out, n_out])
        weights = np.zeros((K, N))

        for k in range(K):
            mu_x = mu[k][input_idx]   # (Dx,)
            mu_y = mu[k][output_idx]  # (Dy,)
            cov_xx = cov[k][np.ix_(input_idx, input_idx)]      # (Dx, Dx)
            cov_xy = cov[k][np.ix_(input_idx, output_idx)]     # (Dx, Dy)
            cov_yx = cov[k][np.ix_(output_idx, input_idx)]     # (Dy, Dx)
            cov_yy = cov[k][np.ix_(output_idx, output_idx)]    # (Dy, Dy)
            inv_cov_xx = np.linalg.inv(cov_xx + 1e-10 * np.eye(len(input_idx)))  # Regularize

            all_covs[k] = cov_yy - cov_yx @ inv_cov_xx @ cov_xy

            diff_x = input_data - mu_x   # (N, Dx)
            # Compute conditional mean for each sample
            all_means[k] = mu_y + (cov_yx @ inv_cov_xx @ diff_x.T).T  # (N, Dy)

            weights[k] = pi[k] * self.multivariate_gaussian(input_data, mu_x, cov_xx) # (N,)

        # Normalize weights
        weights_sum = np.sum(weights, axis=0, keepdims=True) + 1e-12
        weights = weights / weights_sum  # (K, N)
        weighted_pred = np.sum(weights[..., np.newaxis] * all_means, axis=0)  # (N, Dy)
        mode_idx = np.argmax(weights, axis=0)
        mode_pred = all_means[mode_idx, np.arange(N)]  # (N, Dy)

        return mode_pred, weighted_pred, all_means, weights, all_covs
        
    def predict(self, input_data, input_idx, output_idx, method="mean"):
        mode_pred, weights_pred, _, _, _ = self.full_mixture(input_data, input_idx, output_idx)
        if method == "mean":
            return weights_pred
        elif method == "mode":
            return mode_pred
        else:
            raise ValueError("method must be either 'mean' or 'mode'")
    
    def predict_logpdf(self, input_data, output_data, input_idx, output_idx):
        """
        Returns: (N,) logpdf values under the GMR conditional mixture
        """
        N = input_data.shape[0]
        _, _, all_means, weights, all_covs = self.full_mixture(input_data, input_idx, output_idx)
        logpdfs = np.zeros(N) # [N,]
        for i in range(N):
            y = output_data[i]
            comps = []
            for k in range(weights.shape[0]):
                mean = all_means[k, i]
                cov = all_covs[k]
                # Log-pdf of y under component k
                det_cov = np.linalg.det(cov)
                inv_cov = np.linalg.inv(cov)
                norm_const = 1 / np.sqrt((2 * np.pi) ** len(y) * det_cov)
                diff = y - mean
                exp_term = np.exp(-0.5 * diff @ inv_cov @ diff)
                log_pdf = np.log(norm_const * exp_term + 1e-300)  # Prevent log(0)
                comps.append(np.log(weights[k, i] + 1e-12) + log_pdf)
            logpdfs[i] = logsumexp_np(comps)
        return logpdfs

    def sample(self, input_data, input_idx, output_idx, n_samples=1, random_state=None):
        """
        Sample output(s) given input(s) using the GMR conditional mixture.
        input_data: (N, len(input_idx))
        Returns: samples (N, n_samples, len(output_idx))
        """
        if random_state is not None:
            np.random.seed(random_state)
        N = input_data.shape[0]
        _, _, all_means, weights, all_covs = self.full_mixture(input_data, input_idx, output_idx)
        K = weights.shape[0]
        n_out = len(output_idx)
        samples = np.zeros((N, n_samples, n_out))
        for i in range(N):
            # For each sample, sample n_samples from the conditional mixture
            # First, select which component (using mixture weights)
            comp_choices = np.random.choice(K, size=n_samples, p=weights[:, i])
            for j, k in enumerate(comp_choices):
                # Draw from the conditional Gaussian
                samples[i, j] = np.random.multivariate_normal(all_means[k, i], all_covs[k])
        return samples

def logsumexp_np(arr):
        arr = np.array(arr)
        maxa = np.max(arr)
        return maxa + np.log(np.sum(np.exp(arr - maxa)))