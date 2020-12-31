from numpy.core.fromnumeric import nonzero
from numpy.testing._private.utils import print_assert_equal
from ..a_trainer import Trainer
from .svm_model import SVMModel
import numpy as np

class SVMTrainer(Trainer):
    ITERATIONS=10
    EPSILON=1E-15 #add to stuff you divide by that might be 0

    def __init__(self,kernel_func,C):
        super().__init__()
        self.kernel_func=kernel_func
        self.C=C

    def restrict_to_square(self, t, v_0, u):
        t = (np.clip(v_0 + t*u, 0, self.C) - v_0)[1]/u[1]
        return (np.clip(v_0 + t*u, 0, self.C) - v_0)[0]/u[0]

    def train(self,training_dataset,validation_dataset=None):
        X,y=training_dataset.get_X_y()
        X=np.array(X)
        y=np.array(y)
        m=len(X)

        alphas=[]
        neg_counter=0
        pos_counter=0
        for y_i in y:
            if y_i>0:
                pos_counter+=1
            else:
                neg_counter+=1
        for y_i in y:
            if y_i>0:
                alphas.append((pos_counter/(m))*self.C/m)
            else:
                alphas.append((neg_counter/(m))*self.C/m)
        alphas=np.array(alphas)

        K=self.kernel_func.compute(X,X)*y[:,np.newaxis]*y
        for _ in range(SVMTrainer.ITERATIONS):
            for i in range(len(alphas)):
                j=np.random.randint(0,len(alphas))
                while j==i:
                    j=np.random.randint(0,len(alphas))
                v_0=alphas[[i,j]]
                Q=K[[[i,i],[j,j]],[[i,j],[i,j]]]
                k_0=1-np.sum(alphas*K[[i,j]],axis=1)
                u=np.array([-y[j],y[i]])
                t=np.dot(k_0,u)/(np.dot(np.dot(Q,u),u)+SVMTrainer.EPSILON)
                alphas[[i,j]]=v_0+u*self.restrict_to_square(t,v_0,u)
        non_zero_alphas,=np.nonzero(alphas>SVMTrainer.EPSILON)
        b=np.sum((1.0-np.sum(K[non_zero_alphas]*alphas,axis=1))*y[non_zero_alphas])/len(non_zero_alphas)

        return SVMModel(X,y,alphas,b,self.kernel_func)