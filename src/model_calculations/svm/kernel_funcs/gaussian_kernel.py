from ..a_kernel_func import KernelFunc

import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform

class GaussianKernel(KernelFunc):
    def __init__(self, tau):
        super().__init__()
        self.tau=tau

    def compute_gram(self,el): #from https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
        pairwise_dists = squareform(pdist(el, 'sqeuclidean'))
        K = np.exp(-self.tau*pairwise_dists )
        return K

    def compute(self, v, u):
        el=np.concatenate((v,u),axis=0)
        K=self.compute_gram(el) # this is somehow WAY faster than just outright calculating it
        result= K[:len(v),len(v):]
        return result

    def to_string(self):
        return "gaussian with "+ str(self.tau)