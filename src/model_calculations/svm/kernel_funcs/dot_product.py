from ..a_kernel_func import KernelFunc
import numpy as np

class DotProduct(KernelFunc):
    def compute(self, v, u):
        return np.dot(v,u.T)

    def to_string(self):
        return "Dot product"