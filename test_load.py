import numpy as np
import os
import gzip

# compile all arrays into one big one
# also sort of kind of testing

# directory to grab arrays from
directory = "data_collection"
# name of compiled final array
final_file = "Vector_data_full_arr"
# empty array size for ball + car + actions in 1v1
arr = None
# loop over files in directory, load them with numpy and throw them into an array
for f_str in os.listdir(directory):
    file_str = f"{directory}/{f_str}"
    f = np.load(file_str)
    if arr is None:
        arr = f
    else:
        arr = np.vstack((arr, f))

arr = np.expand_dims(arr, axis=0)
f_comp = gzip.GzipFile(f"{directory}/{final_file}_compressed", "w")
np.save(f_comp, arr)
f_comp.close()

f_comp = gzip.GzipFile(f"{directory}/{final_file}_compressed", "r")
arr = np.load(f_comp)
f_comp.close()
# print(arr)
