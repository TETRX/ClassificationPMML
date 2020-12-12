import abc

class Model:
    @abc.abstractmethod
    def predict(self, in_dataset):
        pass
    
    def evaluate(self, test_dataset):
        X,y = test_dataset.get_X_y()
        predicted_y=self.predict(X)
        correct_count=0
        for i in range(len(y)):
            if y[i]==predicted_y[i]:
                correct_count+=1
        return correct_count/len(y)