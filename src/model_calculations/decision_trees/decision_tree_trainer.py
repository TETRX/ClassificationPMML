from ..a_trainer import Trainer
from .decision_tree_node import DecisionTreeNode

class DecisionTreeTrainer(Trainer):
    def __init__(self,alpha):
        self.alpha=alpha

    def train(self,training_dataset,validation_dataset=None):
        tree=DecisionTreeNode(training_dataset,[i for i in range(training_dataset.y-1)])
        tree.build()
        self.prune(validation_dataset,tree)
        return tree

    def prune(self,val_dataset, tree):
        prunes=[] #will append all nodes in the order of pruning
        while not tree.is_leaf():
            prune=tree.best_prune()
            prunes.append(prune[0])
            prune[0].set_prune(True)
        tree.set_prune_to_all(False)
        best_time=0
        size=tree.size()
        best_rate=tree.evaluate(val_dataset)+self.alpha*size
        for i in range(len(prunes)):
            cost=1-tree.evaluate(val_dataset)
            reg=self.alpha*(size-i)
            # print("eval:",eval)
            # print("reg:",reg)
            rate=cost+reg
            if rate<best_rate:
                best_rate=rate
                best_time=i
            prunes[i].set_prune(True)
        tree.set_prune_to_all(False)
        for i in range(best_time):
            prunes[i].set_prune(True)
        