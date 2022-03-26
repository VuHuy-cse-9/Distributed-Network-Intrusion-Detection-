import pandas as pd
import numpy as np

# D:\Tryhard\212-Semester\Parallel Computing\Assignment\Distributed-Network-Intrusion-Detection-\dataset\kddcup1999_10percent.csv
def loadFile(csvPathFile):
    df = pd.read_csv (r(csvPathFile)) # load file using pandas
    npArray = df.to_numpy() #pandas to numpy
    return npArray
