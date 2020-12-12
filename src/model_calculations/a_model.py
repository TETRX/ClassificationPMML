import abc

class Model:
    @abc.abstractmethod
    def predict(self, v):
        pass
    
    def predict_on_all(self,in_dataset):
        predictions=[]
        for v in in_dataset:
            predictions.append(self.predict(v))
        return predictions
            

    def evaluate(self, test_dataset):
        X,y = test_dataset.get_X_y()
        predicted_y=self.predict_on_all(X)
        correct_count=0
        for i in range(len(y)):
            if y[i]==predicted_y[i]:
                correct_count+=1
        return correct_count/len(y)