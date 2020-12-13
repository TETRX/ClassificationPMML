from ..a_model import Model

class SVMModel(Model):
    def __init__(self,w,b,kernel):
        super().__init__()
        self.b=b
        self.w=w
        self.kernel=kernel

    def predict(self,v):
        dot_product=self.kernel.compute(v,self.w)
        result= 1 if dot_product+self.b>0 else -1
        return result