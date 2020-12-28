from ..a_kernel_func import KernelFunc

class DotProduct(KernelFunc):
    def compute(self, v, u):
        return sum(i[0] * i[1] for i in zip(v, u))

    def to_string(self):
        return "Dot product"