from ..a_kernel_func import KernelFunc

import numpy as np

class GaussianKernel(KernelFunc):
    def __init__(self, tau):
        super().__init__()
        self.tau=tau

    def compute(self, v, u):
        result= np.exp(np.sum((v-u[:,np.newaxis])**2,axis=-1)/(2*self.tau**2))
        return result

    def to_string(self):
        return "gaussian with "+ str(self.tau)