from ..data_processing.dataset import Dataset
from ..data_processing.data_reader import DataReader
from ..data_processing.data_divider import DataDivider


from ..model_calculations.svm.kernel_funcs.dot_product import DotProduct
from ..model_calculations.svm.svm_validator import SVMValidator
from ..model_calculations.svm.sequential_optimizer import SequentialOptimizer
from ..model_calculations.svm.kernel_funcs.polynomial_kernel import PolynomialKernel

data_reader=DataReader("data/phishing.data")
dataset=data_reader.read()
data_divider=DataDivider()
training_dataset, validation_dataset, test_dataset= data_divider.divide([0.3,0.3,0.3],dataset)
training_validating_datasets=[(training_dataset,validation_dataset)]


Cs=[5000]
kernel_funcs=[DotProduct(),PolynomialKernel(10)]
optimizer=SequentialOptimizer()

validator=SVMValidator(Cs,kernel_funcs,optimizer)

trainer=validator.get_trainer(training_validating_datasets)

model=trainer.train(training_dataset)
print(model.evaluate(test_dataset))