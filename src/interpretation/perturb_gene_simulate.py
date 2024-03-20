from models import InferenceAutoGenerator
from utils import get_day2_progenitors
import torch
from types import SimpleNamespace
from itertools import product
from tqdm.auto import tqdm
from scipy.io import mmread
from scipy.sparse import csc_matrix
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from collections import Counter
import annoy


device = torch.device('cuda')
NS = 100
num_steps = int(4/0.1)
N_STEPS = 100


def overExpress(x_i,M, N, mono_tfs, neutro_tfs):
    #Get the gene expression for that cell
    if M != 0:
      x_i[mono_tfs] = M
    if N != 0:
      x_i[neutro_tfs] = N
    return x_i


def perturb_gene_simulate(model, mono_tfs, neutro_tfs, countsInVitroCscMatrix, cellcell_score,metadata, clone_data, pca_all):

  day2 = get_day2_progenitors(metadata, clone_data)


  ay = annoy.AnnoyIndex(50,'euclidean')
  ay.load('/content/drive/MyDrive/SrJ/Weinreb/50_20_10.ann')
  with open('/content/drive/MyDrive/SrJ/Weinreb/50_20_10.txt', 'r') as f:
      cy = np.array([line.strip() for line in f])

  count_dict = {}
  for m, n in tqdm(list(product([0,1,2,2.5],[0,-1,-2,-2.5]))):
    cellcell_score = pd.read_csv(f"/content/drive/MyDrive/SrJ/Weinreb/R/cellcellscore/cell_cell_interaction_score_Monocyte_{m}_Neutrophil_{n}.csv").values
    count_df = pd.DataFrame(columns = ['Monocyte','Neutrophil','Other'])
    for _ in tqdm(range(N_STEPS), leave = True):
      x_i = countsInVitroCscMatrix.iloc[day2].copy()
      x_i = overExpress(x_i,m,n,mono_tfs,neutro_tfs)
      x_i = pca_all.transform(x_i.values)
      x_i = torch.tensor(x_i).float().to(device)
      c_i = torch.tensor(np.array(cellcell_score[day2])).float().to(device)

      x_i, _ = model.forward(x_i,c_i)
      x_i = x_i.detach().cpu().numpy()

      yp = []
      for j in range(x_i.shape[0]):
          nn = cy[ay.get_nns_by_vector(x_i[j], 20)]
          nn = Counter(nn).most_common(2)
          label, num = nn[0]
          if len(nn) > 1:
              _, num2 = nn[1]
              if num == num2:  # deal with ties by setting it to the default class
                  label = 'Other'
          yp.append(label)
      yp = Counter(yp)
      count_df = count_df.append(yp, ignore_index = True)
    count_dict[(m,n)] = count_df

  torch.save({"count": count_dict},"/content/drive/MyDrive/SrJ/Weinreb/Interpretation/perterbation_simulation_data_monocyte_up__ModelX_100sim.pt")
  return count_dict

