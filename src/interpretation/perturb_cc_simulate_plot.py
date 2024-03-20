from tqdm.auto import tqdm
from itertools import product
import pandas as pd
import torch
import numpy as np
from collections import Counter
import annoy
from utils import get_day2_progenitors
import matplotlib.pyplot as plt
import seaborn as sns

N_STEPS = 100
device = torch.device('cuda')


def perturb_cc_simulate_plot(model,pca_all, countsInVitroCscMatrix, metadata, clone_data, cc_up_genes, cc_up):

    day2 = get_day2_progenitors(metadata, clone_data)

    count_dict = {}
    for m,n in tqdm(list(product([0],[0,5,-5]))):
        cellcell_score = pd.read_csv(f"/content/drive/MyDrive/SrJ/Weinreb/R/cellcellscore/cell_cell_interaction_score_Monocyte_{0}_Neutrophil_{0}.csv").values
        count_df = pd.DataFrame(columns = ['Monocyte','Neutrophil','Other'])
        for _ in tqdm(range(N_STEPS), leave = True):
            c_i = cellcell_score[day2]
            x_i = countsInVitroCscMatrix.iloc[day2].copy()
            if n != 0:
                x_i[cc_up_genes] = n
                c_i[cc_up] = n
            x_i = pca_all.transform(x_i)
            x_i = torch.tensor(x_i).float().to(device)
            c_i = torch.tensor(np.array(c_i)).float().to(device)


            x_i, _ = model.forward(x_i,c_i)
            x_i = x_i.detach().cpu().numpy()


            ay = annoy.AnnoyIndex(50,'euclidean')
            ay.load('/content/drive/MyDrive/SrJ/Weinreb/50_20_10.ann')
            with open('/content/drive/MyDrive/SrJ/Weinreb/50_20_10.txt', 'r') as f:
                cy = np.array([line.strip() for line in f])

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

    torch.save({"count_dict": count_dict},"perturb_cc_sim_100.pt")

    #Plot
    fig, ax = plt.subplots(1,3,figsize=(12,4),sharey= True)
    sns.boxplot(data = count_dict[0,0]/len(count_dict[0,0]) * 100, ax= ax[0])
    ax[0].set_title("No Perturb")
    sns.boxplot(data = count_dict[0,5]/len(count_dict[0,5]) * 100, ax= ax[1])
    ax[1].set_title("Upregulated (+5)")
    sns.boxplot(data = count_dict[0,-5]/len(count_dict[0,5]) * 100, ax= ax[2])
    ax[2].set_title("Downregulated (-5)")
    plt.show()
