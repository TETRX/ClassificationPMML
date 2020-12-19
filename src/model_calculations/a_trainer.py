import abc

class Trainer():

    @abc.abstractmethod
    def train(self,training_dataset,validation_dataset=None):
        pass