from ..a_kernel_func import KernelFunc
import math

class PolynomialKernel(KernelFunc):
    def __init__(self, p):
        super().__init__()
        self.p=p

    def compute(self, v, u):
        return (sum(i[0] * i[1] for i in zip(v, u))+1)**self.p

    def to_string(self):
        return "polynomial with deg: "+self.p