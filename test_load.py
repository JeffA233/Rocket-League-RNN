import numpy as np
import os

# compile all arrays into one big one
# also sort of kind of testing

# directory to grab arrays from
directory = "data_collection"
# name of compiled final array
final_file = "Vector_data_full_arr"
# empty array size for ball + car + actions in 1v1
arr = np.empty((43, 0))
# loop over files in directory, load them with numpy and throw them into an array
for f_str in os.listdir(directory):
    file_str = f"{directory}/{f_str}"
    f = np.load(file_str)
    arr = np.c_[arr, f]

np.save(f"{directory}/{final_file}", arr)
arr = np.load(f"{directory}/{final_file}.npy")
# print(arr)
