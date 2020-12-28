from ..a_validator import Validator
from .decision_tree_trainer import DecisionTreeTrainer
import math

class DecisionTreeValidator(Validator):
    def __init__(self, alphas):
        super().__init__()
        self.alphas=alphas
        

    def get_trainer(self, training_validating_datasets):
        best_eval=math.inf
        best_alpha=0
        for alpha in self.alphas:
            trainer=DecisionTreeTrainer(alpha)
            total_evaluation=0
            for training_dataset,validating_dataset in training_validating_datasets:
                model=trainer.train(training_dataset,validation_dataset=validating_dataset)
                total_evaluation+=model.evaluate(validating_dataset)
            total_evaluation/=len(training_validating_datasets)
            if total_evaluation>best_eval:
                best_eval=total_evaluation
                best_alpha=alpha
            print("alpha:",alpha)
            print("eval:", total_evaluation)
        return DecisionTreeTrainer(best_alpha)