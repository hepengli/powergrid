import os
import pickle
from os.path import abspath, dirname


def load_dataset(file_path):
    # Load dataset
    # Go up 3 levels from data_loader.py to project root
    dir = dirname(dirname(dirname(abspath(__file__))))
    data_dir = os.path.join(dir, 'data', file_path)
    with open(data_dir, 'rb') as file:
        dataset = pickle.load(file)
    return dataset