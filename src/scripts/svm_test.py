from ..model_calculations.svm.kernel_funcs.gaussian_kernel import GaussianKernel
from ..data_processing.dataset import Dataset
from ..data_processing.data_reader import DataReader
from ..data_processing.data_divider import DataDivider


from ..model_calculations.svm.kernel_funcs.dot_product import DotProduct
from ..model_calculations.svm.svm_validator import SVMValidator
from ..model_calculations.svm.sequential_optimizer import SequentialOptimizer
from ..model_calculations.svm.kernel_funcs.polynomial_kernel import PolynomialKernel


from ..result_processing.result_getter import ResultGetter
from ..result_processing.csv_result_saver import CSVResultSaver

PATH_TO_RESULTS="results/svm_dot_prod.csv"
# PATH_TO_RESULTS="results/svm_polynomials.csv"
# PATH_TO_RESULTS="results/svm_gaussians.csv"

NUM_OF_DIVS=5

data_reader=DataReader("data/phishing.data")
dataset=data_reader.read()
data_divider=DataDivider()
divisions= [data_divider.divide([8,1,1],dataset) for i in range(NUM_OF_DIVS) ]
training_validating_datasets=[(div[0],div[1]) for div in divisions]



Cs=[100,1000,5000,10000,20000]

kernel_funcs=[DotProduct()]
# kernel_funcs=[PolynomialKernel(i+1) for i in range(10)]
# ds=[0.001,0.01,0.1,1.0,10.0]
# kernel_funcs=[GaussianKernel(d) for d in ds]

optimizer=SequentialOptimizer()

validator=SVMValidator(Cs,kernel_funcs,optimizer)

trainer=validator.get_trainer(training_validating_datasets)
average_eval=0
i=0


result_getter=ResultGetter()
results=result_getter.get_results([0.01,0.02,0.03,0.125,0.625,1.0],trainer,data_divider,divisions)
print(results)
result_saver=CSVResultSaver(PATH_TO_RESULTS)
result_saver.save(results)