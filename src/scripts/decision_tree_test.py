from ..data_processing.data_reader import DataReader
from ..data_processing.data_divider import DataDivider


from ..model_calculations.decision_trees.decision_tree_validator import DecisionTreeValidator

from ..result_processing.result_getter import ResultGetter
from ..result_processing.csv_result_saver import CSVResultSaver

NUM_OF_DIVS=5


data_reader=DataReader("data/phishing.data")

dataset=data_reader.read()
data_divider=DataDivider()
divisions= [data_divider.divide([8,1,1],dataset) for i in range(NUM_OF_DIVS) ]
training_validating_datasets=[(div[0],div[1]) for div in divisions]


alphas=[0]

validator=DecisionTreeValidator(alphas)

trainer=validator.get_trainer(training_validating_datasets)
average_eval=0
i=0

result_getter=ResultGetter()
results=result_getter.get_results([0.01,0.02,0.03,0.125,0.625,1.0],trainer,data_divider,divisions)
print(results)
result_saver=CSVResultSaver("results/results_decision_trees.csv")
result_saver.save(results)