import os
import numpy as np

def create_directory(dir):
    try:
        parent_dir = "C:"
        path = os.path.join(parent_dir, dir)
        os.mkdir(path)
        print(" ")
        print("Directory created")
    except:
        print("An error occured")

def load_directory_info(directory, final_file):
    arr = None
    
    for f_str in os.listdir(directory):
        if f_str is f"{directory}/{final_file}_compressed":
            continue
            # maybe not necessary to check to skip (was for previous functionality)
        file_str = f"{directory}/{f_str}"
        f = np.load(file_str)
        if arr is None:
            arr = f
        else:
            arr = np.vstack((arr, f))
    return arr