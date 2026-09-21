import numpy as np

# Load using full file path
data_file = "/zhome/07/c/168354/courses/battery-predictions-using-dl/project/src/data/data_m1a_efc0.npy"
data = np.load(data_file, allow_pickle=True)
print(f"Loaded file: {data_file}")
print(data.shape)
print(data)
