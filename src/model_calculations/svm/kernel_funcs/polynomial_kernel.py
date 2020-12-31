from ..a_kernel_func import KernelFunc
import numpy as np

class PolynomialKernel(KernelFunc):
    def __init__(self, p):
        super().__init__()
        self.p=p

    def compute(self, v, u):
        return (np.dot(v,u.T)+1)**self.p

    def to_string(self):
        return "polynomial with deg: "+str(self.p)