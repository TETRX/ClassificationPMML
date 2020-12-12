import abc

class Optimizer:
    @abc.abstractmethod
    def optimize(self, y,x,C,kernel_func):
        pass 