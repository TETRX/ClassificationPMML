from ..a_trainer import Trainer
from .svm_model import SVMModel

class SVMTrainer(Trainer):
    def __init__(self,kernel_func,C,optimizer):
        super().__init__()
        self.kernel_func=kernel_func
        self.C=C
        self.optimizer=optimizer

    def train(self,training_dataset):
        X,y=training_dataset.get_X_y()
        alphas=self.optimizer.optimize(X,y,self.C,self.kernel_func)
        w=[0 for i in range(len(X[0]))]
        for i in range(len(alphas)): #w=sum(alpha_iy^ix^i)
            sum_elem=[alphas[i]*y[i]*x_elem for x_elem in X[i]]
            for j in range(len(w)):
                w[j]+=sum_elem[j]
        b=y[0]-self.kernel_func.compute(w,X[0])
        return SVMModel(w,b,self.kernel_func)
