import numpy as np
import pandas as pd

import os
from pathlib import Path
import math

class Generator:

    # Initilise the class
    #   @ filePaths: Array of the file paths to each table you want to open
    def __init__(self, filePaths):
        for filePath in filePaths:
            # Get strings of the path, file name, and ext
            (path, file) = os.path.split(filePath)
            (name, ext) = os.path.splitext(file)
            # Check whether path is absolute or relative
            if os.path.isabs(path):
                filePath = Path(path, file)
            else:
                filePath = Path(os.path.abspath('.'), filePath)
            # Check path exists and includes file name
            if not os.path.exists(filePath):
                raise ValueError(f'Path to the file "{filePath}" does not exist.')
            elif not os.path.isfile(filePath):
                raise ValueError(f'File does not exist in: "{filePath}"')
            # Import tables and store @todo: Use the different import functions for different file extensions
            self.mTables[name] = pd.read_csv(filePath)
        return

    # Combine all the tables that have been opened
    #   @ shuffle: Shuffle the table after it has been combined
    def combineTables(self, *, shuffle=False):
        # Initialise empty array
        tables = []
        # Iterate through tables and store reference to each table in array
        for table in self.mTables:
            tables.append(self.mTables[table])
        # Concat tables
        self.mCTable = pd.concat(tables, copy=False)
        # Shuffle table
        if shuffle:
            self.mCTable = self.mCTable.sample(frac=1)
            self.mCTable.reset_index(inplace=True, drop=True)
        return

    # Export the combined table as a csv file
    #   @ path: The string path to the folder to export the files to
    #   @ split: How to split the data for each file. Tuple is in order of Training, Testing, Validation
    def exportCTable(self, path="", *, split=(0.8, 0.9, 1)):
        # Check there is a combined table
        if self.mCTable is None:
            raise ValueError("No table has been combined to be exported.")
        # Get the length of combined table
        length = len(self.mCTable)
        # Export to csv:
        # "Training data"
        self.mCTable[0:
                     math.floor(length*split[0])].to_csv(Path(path, "Training.csv"), index=False)
        # "Testing data"
        self.mCTable[math.floor(length*split[0]):
                     math.floor(length*split[1])].to_csv(Path(path, "Testing.csv"), index=False)
        # "Validation data"
        self.mCTable[math.floor(length*split[1]):].to_csv(Path(path, "Validation.csv"), index=False)
        return

    mTables = {}
    mCTable = None


if __name__ == "__main__":
    gen = Generator(["C:\Projects\FakeNewsML\Datasets\Fake.csv", "True.csv"])
    gen.mTables["True"]["label"] = 1
    gen.mTables["Fake"]["label"] = 0
    gen.combineTables(shuffle=True)
    print(gen.mCTable.loc[((gen.mCTable.label==1) & (gen.mCTable.subject=="US_News")),:])
    # gen.exportCTable()