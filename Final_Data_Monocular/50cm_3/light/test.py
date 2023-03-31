import pandas as pd
import numpy as np

# Load the CSV file using pandas
df = pd.read_csv('~/Desktop/Depth_Estimation/Final_Data_Monocular/50cm_3/light/depth_cylinder.csv')

# Convert the data to a numpy array
array = df.to_numpy()

# Find the average of the array
average = np.mean(array)

# Print the average of the array
print("The average of the array is:", average)