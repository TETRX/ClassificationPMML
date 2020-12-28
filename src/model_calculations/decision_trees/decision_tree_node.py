from ..a_model import Model
from ...data_processing.dataset import Dataset
from math import inf

class DecisionTreeNode(Model):
    def __init__(self,data, available_attributes):
        self.X, self.y=data.get_X_y()
        self.available_attributes=available_attributes
        self.attribute=None
        count=[0,0]
        for ans in self.y:
            count[1 if ans==0 else 0]+=1
        self.ret= -1 if count[0]>count[1] else 1
        self.error=count[1] if count[0]>count[1] else count[0] #wrongly evaluated entries
        self.children=[]
        self.pruned=False


    def is_leaf(self):
        if self.pruned:
            return True
        if len(self.available_attributes)==0 or len(self.X)<5:
            return True
        return False

    def rate_attribute(self,attribute):
        counter=[[0,0],[0,0],[0,0]]
        for i in range(len(self.X)):
            counter[self.X[i][attribute]+1][1 if self.y[i]==-1 else 0]+=1
        makes_sense=False
        for comparison in counter:
            if comparison[(self.ret+1)//2]<comparison[(-self.ret+1)//2]: #check if at least one entry has been reclassified
                makes_sense=True 
        if makes_sense:
            return sum(min(count[0],count[1]) for count in counter)
        else:
            return inf

    def make_children(self, attribute):
        attributes=[attribute1 for attribute1 in self.available_attributes if attribute1!=attribute]

        data=[[],[],[]]
        for i in range(len(self.X)):
            data[self.X[i][attribute]+1].append(self.X[i]+[self.y[i]])
        datasets=[Dataset(data_lines) for data_lines in data]
        self.children=[]
        for dataset in datasets:
            self.children.append(DecisionTreeNode(dataset,attributes))

    def build(self):
        if not self.is_leaf():
            best_attribute=self.available_attributes[0]
            best_attribute_rate=self.rate_attribute(best_attribute)
            for attribute in self.available_attributes:
                rate=self.rate_attribute(attribute)
                if rate<best_attribute_rate:
                    best_attribute=attribute
                    best_attribute_rate=rate
            if best_attribute_rate==inf:
                self.available_attributes=[]
            else:
                self.attribute=best_attribute
                self.make_children(best_attribute)
                for child in self.children:
                    child.build()

    def predict(self, v):
        if not self.is_leaf():
            return self.children[v[self.attribute]].predict(v)
        else:
            return self.ret

    def set_prune(self,value):
        self.pruned=value

    def set_prune_to_all(self,value):
        for child in self.children:
            child.set_prune_to_all(value)
        self.pruned=value

    def best_prune(self):
        all_children_leaves=True
        for child in self.children:
            all_children_leaves=all_children_leaves and child.is_leaf()
        if all_children_leaves:
            all_children_error=sum(child.error for child in self.children)
            return (self, self.error-all_children_error)
        else:
            candidates=[]
            for child in self.children:
                if not child.is_leaf():
                    candidates.append(child.best_prune())
            best_cand=candidates[0][0]
            best_cand_rate=candidates[0][1]
            for candidate in candidates:
                rate=candidate[1]
                if rate<best_cand_rate:
                    best_cand_rate=rate
                    best_cand=candidate
            return (best_cand,best_cand_rate)
    
    def size(self):
        if self.is_leaf():
            return 1
        return sum(child.size() for child in self.children)