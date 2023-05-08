import os
import pandas as pd

def create_folder(path):
    if not os.path.isdir(path + "/"):
        os.mkdir(path + "/")



def read_csv(path, sep=",", header=None):
    if os.path.exists(path):
        return pd.read_csv(path, sep=sep, header=header)
    raise FileNotFoundError("File ", path, " not found")
