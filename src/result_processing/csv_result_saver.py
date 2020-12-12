import csv
import os

class CSVResultSaver:
    def __init__(self, path_to_file):
        self.path_to_file=os.path.abspath(path_to_file)

    def save(self,results):
        with open(self.path_to_file,'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in results.items():
                writer.writerow([str(key),str(value)])