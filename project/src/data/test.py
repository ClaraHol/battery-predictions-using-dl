import numpy as np

# Load using full file path
data_file = r"c:\Users\Anton Espholm\Documents\Noter 2\battery-predictions-using-dl\project\src\data\data_m1a_efc0.npy"
data = np.load(data_file, allow_pickle=True)
print(f"Loaded file: {data_file}")
print(data.shape)
