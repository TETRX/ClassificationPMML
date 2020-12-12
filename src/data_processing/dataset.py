from numpy import matrix
import math

class Dataset:
    def __init__(self, data_lines):
        self.matrix=matrix(data_lines)
        self.x=len(data_lines)
        self.y=len(data_lines[0])

    def apply_functions(self, functions):
        new_lines=[]
        for row in self.matrix:
            new_line=[]
            for i in range(self.y):
                new_line.append(functions[i](row[0,i]))
            new_lines.append(new_line)
        return Dataset(new_lines)

    def columnwise_mean(self):
        return [ self.matrix.mean(0)[0,i] for i in range(self.y)]

    def columnwise_std(self):
        return [self.matrix.std(0)[0,i] for i in range(self.y)]

    def data_lines(self):
        return self.matrix.tolist()

    def get_X_y(self):
        lists=self.matrix.tolist()
        X=[]
        y=[]
        for list in lists:
            X.append(list[:-1])
            y.append(list[-1])
        return (X,y)