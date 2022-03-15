import pandas as pd
import numpy as np

# D:\Tryhard\212-Semester\Parallel Computing\Assignment\Distributed-Network-Intrusion-Detection-\dataset\kddcup1999_10percent.csv
df = pd.read_csv (r'dataset\kddcup1999_10percent.csv')

npArray = df.to_numpy()
print(npArray)