from ..data_processing.dataset import Dataset
from ..data_processing.data_reader import DataReader
from ..data_processing.data_divider import DataDivider


from ..model_calculations.decision_trees.decision_tree_validator import DecisionTreeValidator

data_reader=DataReader("data/phishing.data")
dataset=data_reader.read()
data_divider=DataDivider()
training_dataset, validation_dataset, test_dataset= data_divider.divide([1,1,1],dataset)
training_validating_datasets=[(training_dataset,validation_dataset)]


alphas=[0]

validator=DecisionTreeValidator(alphas)

trainer=validator.get_trainer(training_validating_datasets)

model=trainer.train(training_dataset,validation_dataset=validation_dataset)
print(model.evaluate(test_dataset))