# Modified from Ashwood Z.C. et al 2022
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
# Import useful functions from ssm package
from ssm.util import ensure_args_are_lists
from ssm.optimizers import adam, bfgs, rmsprop, sgd
import ssm.stats as stats
from scipy.special import expit
import matplotlib.pyplot as plt


class Glm:
    def __init__(self,M,C) -> None:
        """
        M is number of covarities we wish to have, aka the input size
        C is number of classes in categorical observations, e.g. left/right
        """
        self.M = M 
        self.C = C
        
        self.Wk = npr.randn(1,C-1,M+1)
        self.loglikelihood_train = None
        
    def fit(self,inputs,datas):
        self.fit_glm(datas, inputs, masks=None, tags=None)
        self.loglikelihood_train = self.log_marginal(datas,inputs,None,None)
        
    @property
    def params(self):
        return self.Wk
    
    @params.setter
    def params(self,value):
        self.Wk = value
        
    def log_prior(self):
        return 0
    
    # Calculate time dependent logits - output is matrix of size Tx1xC
    # Input is size TxM
    def calculate_logits(self,input):
        # Update input to include offset term:
        input = np.append(input, np.ones((input.shape[0], 1)), axis=1)
        # Add additional row (of zeros) to second dimension of self.Wk        
        Wk = self.append_zeros(self.Wk)
        # Input effect; transpose so that output has dims TxKxC
        time_dependent_logits = np.transpose(np.dot(Wk, input.T), (2, 0, 1))
        time_dependent_logits = time_dependent_logits - logsumexp(
            time_dependent_logits, axis=2, keepdims=True)
        return time_dependent_logits
    
    # Calculate log-likelihood of observed data
    def log_likelihoods(self, data, input, mask, tag):
        time_dependent_logits = self.calculate_logits(input)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.categorical_logpdf(data[:, None, :],
                                        time_dependent_logits[:, :, None, :],
                                        mask=mask[:, None, :])
        
    # log marginal likelihood of data
    @ensure_args_are_lists
    def log_marginal(self, datas, inputs, masks, tags):
        elbo = self.log_prior()
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            lls = self.log_likelihoods(data, input, mask, tag)
            elbo += np.sum(lls)
        return elbo
    
    @ensure_args_are_lists
    def fit_glm(self,
                datas,
                inputs,
                masks,
                tags,
                num_iters=1000,
                optimizer="bfgs",
                **kwargs):
        optimizer = dict(adam=adam, bfgs=bfgs, rmsprop=rmsprop,
                         sgd=sgd)[optimizer]

        def _objective(params, itr):
            self.params = params
            obj = self.log_marginal(datas, inputs, masks, tags)
            return -obj

        self.params = optimizer(_objective,
                                self.params,
                                num_iters=num_iters,
                                **kwargs)
    
    # Append column of zeros to weights matrix in appropriate location
    @staticmethod
    def append_zeros(weights) -> np.ndarray:
        weights_tranpose = np.transpose(weights, (1, 0, 2))
        weights = np.transpose(
            np.vstack([
                weights_tranpose,
                np.zeros((1, weights_tranpose.shape[1], weights_tranpose.shape[2]))
            ]), (1, 0, 2))
        return weights
    
    def get_prob_right(self,inpt,prev_c,wsls) -> np.ndarray:
        min_val_stim = np.min(inpt[:, 0])
        max_val_stim = np.max(inpt[:, 0])
        stim_vals = np.arange(min_val_stim, max_val_stim, 0.05)
        
        # create input matrix - cols are stim, pc, wsls, bias
        x = np.array([
            stim_vals,
            np.repeat(prev_c, len(stim_vals)),
            np.repeat(wsls, len(stim_vals)),
            np.repeat(1, len(stim_vals))
        ]).T
        wx = np.matmul(x, -self.params[0].T)
        return stim_vals, expit(wx)
    
    def plot_input_vectors(self,labels_for_plot:list=None) -> plt.figure:
        weights = self.append_zeros(self.Wk)
        
        if labels_for_plot is None:
            labels_for_plot = ['Stimulus', 'Past Choice', 'Bias']
        
        K = weights.shape[0]
        K_prime = weights.shape[1]
        M = weights.shape[2] - 1
        
        fig = plt.figure(figsize=(8,9),dpi=80,facecolor='w',edgecolor='k')
        plt.subplots_adjust(left=0.15,
                            bottom=0.27,
                            right=0.95,
                            top=0.95,
                            wspace=0.3,
                            hspace=0.3)
        
        for j in range(K):
            for k in range(K_prime - 1):
                # plt.subplot(K, K_prime, 1+j*K_prime+k)
                plt.plot(range(M + 1), - weights[j][k], marker='o')
                plt.plot(range(-1, M + 2), np.repeat(0, M + 3), 'k', alpha=0.2)
                plt.axhline(y=0, color="k", alpha=0.5, ls="--")

                plt.xticks(list(range(0, len(labels_for_plot))),
                        labels_for_plot,
                        rotation='90',
                        fontsize=12)

                plt.ylim((-3, 6))

        fig.text(0.04,
                0.5,
                "Weight",
                ha="center",
                va="center",
                rotation=90,
                fontsize=15)
        fig.suptitle(f"GLM Weights: {self.loglikelihood_train}", y=0.99, fontsize=14)
        return fig