import abc

class Validator():

    @abc.abstractmethod
    def get_trainer(self, training_dataset, validating_dataset):
        pass