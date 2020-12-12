from .dataset import Dataset
import random

class DataDivider:
    def divide(self,proportions,dataset ):
        list_=dataset.data_lines()
        random.shuffle(list_)
        sum_of_proportions=0
        for prop in proportions:
            sum_of_proportions+=prop
        datasets=[]
        curr_start=0
        for prop in proportions:
            num_of_elem=int((prop*dataset.x)//sum_of_proportions)
            if num_of_elem==0:
                datasets.append([])
            new_list=list_[curr_start:curr_start+num_of_elem]
            datasets.append(new_list)
            curr_start+=num_of_elem
        return tuple([Dataset(list__) for list__ in datasets])

    def get_first(self,proportions,dataset):
        list_=dataset.data_lines()
        random.shuffle(list_)
        sum_of_proportions=0
        for prop in proportions:
            sum_of_proportions+=prop
        prop=proportions[0]
        num_of_elem=int((prop*dataset.x)//sum_of_proportions)
        new_list=list_[0:num_of_elem]
        return Dataset(new_list)