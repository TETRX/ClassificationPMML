from ..a_kernel_func import KernelFunc
import numpy as np

class DotProduct(KernelFunc):
    def compute(self, v, u):
        return np.dot(v,u.T)

    def compute_gram(self, el):
        return self.compute(el,el)
    
    def to_string(self):
        return "Dot product"