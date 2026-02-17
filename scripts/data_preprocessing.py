import pandas as pd

"""
load data and preprocess
"""
#---load the data
def load_data(data_path):
    data = pd.read_csv(data_path)

    return data

#---preprocess the data
def pre_process_data(data):
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    return data

