from ..a_kernel_func import KernelFunc
import math

class GaussianKernel(KernelFunc):
    def __init__(self, tau):
        super().__init__()
        self.tau=tau

    def compute(self, v, u):
        return math.exp(sum((i[0]-i[1])**2 for i in zip(v, u))/(2*self.tau**2))