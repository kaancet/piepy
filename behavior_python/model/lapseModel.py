# Modified from Ashwood Z.C. et al 2022

from warnings import warn
import autograd.numpy as np
import autograd.numpy.random as npr
import ssm.stats as stats
from autograd import grad, hessian
from autograd.misc import flatten
from scipy.optimize import minimize
# Import generic useful functions from ssm package
from ssm.util import ensure_args_are_lists


class LapseModel:
    def __init__(self,M,num_lapse_params, include_bias:bool=True) -> None:
        self.M = M # M is
        self.include_bias = include_bias
        # Parameters linking input to state distribution
        if self.include_bias:
            self.W = 2 * npr.randn(M + 1)
        else:
            self.W = 2 * npr.randn(M)
        self.num_lapse_params = num_lapse_params
        self.lapse_params = self.init_lapse_rates(self.num_lapse_params)
    
    @staticmethod 
    def init_lapse_rates(num_lapse_params) -> np.ndarray:
        """ Lapse rates - initialize gamma and lambda lapse rates as 0.05m withm some noise
        Ensure that gamma and lambda parameters are greater than or equal to 0 and less than or equal to 1
        """
        gamma = np.maximum(0.05 + 0.03 * npr.rand(1)[0], 0)
        gamma = np.minimum(gamma, 1)
        if num_lapse_params == 2:
            lamb = np.maximum(0.05 + 0.03 * npr.rand(1)[0], 0)
            lamb = np.minimum(lamb, 1)
            lapse_params = np.array([gamma, lamb])
        else:
            lapse_params = np.array([gamma])
            
        return lapse_params
        
    @property
    def params(self) -> list:
        return [self.W, self.lapse_params]

    @params.setter
    def params(self, value):
        self.W = value[0]
        self.lapse_params = value[1]

    def log_prior(self) -> int:
        return 0
    
    def calculate_pr_lapse(self) -> float:
        """ Returns the lapse rate probability """
        if self.num_lapse_params == 2:
            pr_lapse = self.lapse_params[0] + self.lapse_params[1]
        else:
            pr_lapse = 2 * self.lapse_params[0]
        return pr_lapse
    
    def calculate_pr_right(self, input:np.ndarray) -> float:
        """ Calculate probability right at each time step
            y=1 is right:
            p(y=1) = gamma + (1-lambda-gamma)(1/(1+e^(-wx)))
        """
        # Update input to include offset term:
        if self.include_bias:
            input = np.append(input, np.ones((input.shape[0], 1)), axis=1)
        logits = np.dot(self.W, input.T)
        softmax = np.exp(logits) / (1 + np.exp(logits))
        if self.num_lapse_params == 2:
            prob_right = self.lapse_params[0] + (
                    1 - self.lapse_params[0] - self.lapse_params[1]) * softmax
        else:
            prob_right = self.lapse_params[0] + (
                    1 - 2 * self.lapse_params[0]) * softmax
        return prob_right, softmax
    
    def calculate_logits(self, input:np.ndarray) -> np.ndarray:
        """ Calculate time dependent logits - output is matrix of size Tx2,
            with pr(R) in 2nd column
            Input is size TxM
        """
        prob_right, _ = self.calculate_pr_right(input)
        assert (max(prob_right) <= 1) or (
                min(
                    prob_right) >= 0), 'At least one of the probabilities is '\
                                       'not between 0 and 1'
        # Now calculate prob_left
        prob_left = 1 - prob_right
        # Calculate logits - array of size Tx2 with log(prob_left) as first
        # column and log(prob_right) as second column
        time_dependent_logits = np.transpose(
            np.vstack((np.log(prob_left), np.log(prob_right))))
        # Add in lapse parameters
        return time_dependent_logits
    
    def log_likelihoods(self, data, input, mask, tag) -> np.ndarray:
        """ Calculate log-likelihood of observed data: 
        LL = sum_{i=0}^{T}(y_{i}log(p_{i}) + (1-y_{i})log(1-p_{i})) 
        where y_{i} is a one-hot vector of the data
        """
        time_dependent_logits = self.calculate_logits(input)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.categorical_logpdf(data[:, None, :],
                                        time_dependent_logits[:, None,
                                        None, :],
                                        mask=mask[:, None, :])
        
    def sample(self, input, tag=None, with_noise=True):
        """ Sample both a state and a choice at each time point
        """
        T = input.shape[0]
        data_sample = np.zeros(T, dtype='int')
        z_sample = np.zeros(T, dtype='int')
        pr_right, softmax = self.calculate_pr_right(
            input)  # vectors of length T
        pr_lapse = self.calculate_pr_lapse()
        for t in range(T):
            z_sample[t] = npr.choice(2, p=[(1 - pr_lapse), pr_lapse])
            if z_sample[t] == 0:
                data_sample[t] = npr.choice(2,
                                            p=[(1 - softmax[t]), softmax[t]])
            else:  # indicates a lapse
                if self.num_lapse_params == 1:
                    data_sample[t] = npr.choice(2, p=[0.5, 0.5])
                else:
                    lapse_pr_right = self.lapse_params[0] / (
                            self.lapse_params[0] + self.lapse_params[1])
                    data_sample[t] = npr.choice(2,
                                                p=[(1 - lapse_pr_right),
                                                   lapse_pr_right])
        data_sample = np.expand_dims(data_sample, axis=1)
        return z_sample, data_sample
    
    @ensure_args_are_lists
    def log_marginal(self, datas, inputs, masks, tags):
        elbo = self.log_prior()
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            lls = self.log_likelihoods(data, input, mask, tag)
            elbo += np.sum(lls)
        return elbo

    @ensure_args_are_lists
    def fit(self,
            datas,
            inputs,
            masks,
            tags,
            optimizer="s",
            num_iters=1000,
            **kwargs):

        # define optimization target
        def _objective(params, itr):
            self.params = params
            obj = self.log_marginal(datas, inputs, masks, tags)
            return -obj

        # Update params as output of optimizer
        self.params, self.hessian = self.minimize_loss(_objective,
                                                  num_iters=num_iters,
                                                  **kwargs)
        
    def minimize_loss(self,loss,verbose:bool=False, num_iters:int=1000) -> np.ndarray:
        # Flatten the loss
        _x0, unflatten = flatten(self.params)
        _objective = lambda x_flat, itr: loss(unflatten(x_flat), itr)
        
        # Specify callback for fitting
        itr = [0]
        
        def callback(x_flat):
            itr[0] += 1
            print("Iteration {} loss: {:.3f}".format(itr[0],
                                                    loss(unflatten(x_flat), -1)))
            print("Grad: ")
            grad_to_print = grad(_objective)(x_flat, -1)
            print(grad_to_print)
        
        # Bounds
        N = self.params[0].shape[0]
        if self.num_lapse_params == 2:
            bounds = [(-10, 10) for i in range(N + 2)]
            bounds[N] = (0, 0.5)
            bounds[N + 1] = (0, 0.5)
        else:
            bounds = [(-10, 10) for i in range(N + 1)]
            bounds[N] = (0, 0.5)
            
        # Call the optimizer
        result = minimize(_objective,
                        _x0,
                        args=(-1,),
                        jac=grad(_objective),
                        method="SLSQP",
                        bounds=bounds,
                        callback=callback if verbose else None,
                        options=dict(maxiter=num_iters, disp=verbose))
        
        if verbose:
            print("Optimization completed with message: \n{}".format(
                result.message))

        if not result.success:
            warn("Optimization failed with message:\n{}".format(result.message))

        # Get hessian:
        autograd_hessian = hessian(_objective)
        hess_to_return = autograd_hessian(result.x, -1)
        return unflatten(result.x), hess_to_return

    @staticmethod
    def get_parmax(i,M):
        if i <= M:
            return 10
        else:
            return 1
    
    @staticmethod 
    def get_parmin(i, M):
        if i <= M:
            return -10
        else:
            return 0

    @staticmethod
    def get_parstart(i, M):
        if i <= M:
            return 2 * npr.randn(1)
        else:
            gamma = np.maximum(0.05 + 0.03 * npr.rand(1), 0)
            gamma = np.minimum(gamma, 1)
            return gamma
        
    # Reshape hessian and calculate its inverse
    @staticmethod
    def calculate_std(hessian):
        # Calculate inverse of Hessian (this is what we will actually use to
        # calculate variance)
        inv_hessian = np.linalg.inv(hessian)
        # Take diagonal elements and calculate square root
        std_dev = np.sqrt(np.diag(inv_hessian))
        return std_dev