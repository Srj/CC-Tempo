import numpy as np
import random
import torch
import os

def get_day2_progenitors(metadata, clone_data):
    #get the cells from day 4 & 6
    day4_6 = metadata.loc[metadata['Time point'] > 3]
    day4_6_neutrophil = day4_6.loc[(day4_6['Annotation'] == 'Neutrophil')]
    day4_6_monocyte = day4_6.loc[(day4_6['Annotation'] == 'Monocyte')]

    #day4_6 contains only cells from day 4 & 6 that are either Neutrophil or Monocyte
    day4_6 = day4_6.index[(day4_6['Annotation'] == 'Neutrophil')
                            | (day4_6['Annotation'] == 'Monocyte')]

    #index of those clones who have Neu/Mon cells at day 4 or 6
    clone_index = np.reshape(
            np.array(np.sum(clone_data[day4_6], axis=0)), 5864) > 0

    #index of those cells who belongs to clones who have Neu/Mon cells at day 4 or 6
    cell_index = np.reshape(
            np.array(np.sum(clone_data[:, clone_index], axis=1)), 130887) > 0

    #get metadata of cells filtered above
    clone_metadata = metadata.loc[cell_index]

    #get indices of those day 2 cells who belongs to the clones who have Neu/Mon cells at day 4 or 6
    day2 = clone_metadata.index[clone_metadata['Time point'] < 3].tolist()
    return day2

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")