from numpy.lib.shape_base import _make_along_axis_idx
from ..a_model import Model
import numpy as np

class SVMModel(Model):
    def __init__(self,X,y,alphas,b,kernel):
        super().__init__()
        self.b=b
        self.X=X
        self.y=y
        self.alphas=alphas
        self.kernel=kernel

    def predict(self,v):
        dot_product=np.sum(self.kernel.compute(v,self.X)*self.y*self.alphas)
        result= 1 if dot_product+self.b>0 else -1
        return result

    def predict_on_all(self, X):
        X=np.array(X)
        dot_product=np.sum(self.kernel.compute(X,self.X)*self.y*self.alphas,axis=1)
        result=[]
        for i in dot_product:
            if i+self.b>0:
                result.append(1)
            else:
                result.append(-1)
        return result