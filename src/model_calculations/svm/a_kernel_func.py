import abc

class KernelFunc():
    @abc.abstractmethod
    def compute(self, v, u):
        pass