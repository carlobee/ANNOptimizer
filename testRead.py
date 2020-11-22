import pandas as pd
import numpy as np

# read values from file and append to arrays
data = pd.read_csv("1in_cubic.txt", header=None, delim_whitespace=True)
        
input_values_x = data[0].tolist()
target_values_y = data[1].tolist()
    
arr = np.empty(len(data.index))
    
print(arr)