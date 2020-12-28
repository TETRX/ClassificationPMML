from ..a_validator import Validator
from .svm_trainer import SVMTrainer
import math

class SVMValidator(Validator):
    def __init__(self, Cs, kernel_funcs, optimizer):
        super().__init__()
        self.Cs=Cs
        self.kernel_funcs=kernel_funcs
        self.optimizer=optimizer

    def get_trainer(self, training_validating_datasets):
        best_eval=math.inf
        best_C=0
        best_kernel=0
        for C in self.Cs:
            for kernel_func in self.kernel_funcs:
                try:
                    trainer=SVMTrainer(kernel_func,C,self.optimizer)
                    total_evaluation=0
                    for training_dataset,validating_dataset in training_validating_datasets:
                        model=trainer.train(training_dataset)
                        total_evaluation+=model.evaluate(validating_dataset)
                    total_evaluation/=len(training_validating_datasets)
                    if total_evaluation<best_eval:
                        best_eval=total_evaluation
                        best_C=C
                        best_kernel=kernel_func
                    print("C:",C)
                    print("kernel:", kernel_func.to_string())
                    print("eval:", total_evaluation)
                except:
                    continue
        return SVMTrainer(best_kernel,best_C,self.optimizer)