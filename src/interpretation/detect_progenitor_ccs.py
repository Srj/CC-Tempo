from utils import set_seed, get_day2_progenitors
from models import InferenceAutoGenerator,WrapperModelc
from cell_classifier import CellClassifier
import torch
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
import annoy
from captum.attr import GradientShap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_importances( importances, title=None):
  fig,ax = plt.subplots(2,1,figsize=(12,8))
  cellcell_score = pd.read_csv(f"/content/drive/MyDrive/SrJ/Weinreb/R/cellcellscore/cell_cell_interaction_score_Monocyte_{0}_Neutrophil_{0}.csv")
  sns.barplot(importances[:,:29],alpha=0.8, color='#4260f5',ax = ax[0])
  ax[0].set_xticks(range(29),cellcell_score.columns.tolist()[:29], rotation=90)
  sns.barplot(importances[:,29:],alpha=0.8, color='#4260f5', ax = ax[1])
  ax[1].set_xticks(range(29),cellcell_score.columns.tolist()[:29], rotation=90)
  # plt.savefig(f'ig/{ID}.jpg', dpi=300)
  plt.title(title)
  plt.tight_layout()
  plt.show()


def progenitor_ccs_detection(model, countsInVitroCscMatrix, cellcell_score,metadata, clone_data, xp, pca_all):
    set_seed()
    device = torch.device('cuda')
    NS = 100
    num_steps = int(4/0.1)

    day2 = get_day2_progenitors(metadata, clone_data)
    ay = annoy.AnnoyIndex(50,'euclidean')
    ay.load('/content/drive/MyDrive/SrJ/Weinreb/50_20_10.ann')
    with open('/content/drive/MyDrive/SrJ/Weinreb/50_20_10.txt', 'r') as f:
        cy = np.array([line.strip() for line in f])


    classifier = CellClassifier(50+58)
    checkpoint = torch.load("/content/drive/MyDrive/SrJ/Weinreb/Classifier/classifier.pt", map_location = device)
    classifier.load_state_dict(checkpoint)
    classifier.to(device)

    x_i = torch.tensor(xp[np.array(day2)], dtype=torch.float32).to(device)
    c_i = torch.tensor(np.array(cellcell_score[np.array(day2)])).float().to(device)

    wrapper = WrapperModelc(model,classifier, pca_all).to(device)
    gs = GradientShap(wrapper)

    baselines = cellcell_score[day2].mean(0).reshape(1,-1)

    for target, title in zip([0,1,2],[ 'Monocyte', 'Neutrophil', 'Others']):
        x_i = torch.tensor(countsInVitroCscMatrix.iloc[np.array(day2)].values, dtype=torch.float32).to(device)
        c_i = torch.tensor(np.array(cellcell_score[np.array(day2)])).float().to(device)
        c_i = c_i.requires_grad_()
        x_i = x_i.requires_grad_()
        attr_mono = gs.attribute(c_i,baselines = torch.tensor(baselines).float().to(device), target = target,
                            additional_forward_args =x_i)

        visualize_importances(attr_mono.detach().cpu().numpy(), title = title)


    

        

