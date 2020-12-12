import os
from .dataset import Dataset
#tested manually

class DataReader:
    def __init__(self,path_to_file):
        self.path_to_file=os.path.abspath(path_to_file)


    def read(self):
        num_lines=[]

        file = open(self.path_to_file, 'r') 
        lines = file.readlines()
        for line in lines:
            num_lines.append([float(num) for num in line.split()])
        return Dataset(num_lines)