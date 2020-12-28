import abc

class KernelFunc():
    @abc.abstractmethod
    def compute(self, v, u):
        pass

    @abc.abstractmethod
    def to_string(self):
        pass