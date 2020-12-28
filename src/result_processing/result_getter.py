class ResultGetter:

    def get_results(self,fractions,trainer,data_divider,divisions):
        result={}
        for fraction in fractions:
            result[fraction]=0
        for training_dataset,validating_dataset,testing_dataset in divisions:
            for fraction in fractions:
                frac_train=data_divider.get_first([fraction,1],training_dataset)
                model=trainer.train(frac_train,validation_dataset=validating_dataset)
                result[fraction]+=model.evaluate(testing_dataset)
        for fraction in fractions:
            result[fraction]/=len(divisions)
        return result