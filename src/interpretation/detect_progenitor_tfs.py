from utils import set_seed, get_day2_progenitors
from models import InferenceAutoGenerator,WrapperModelx
import torch
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
import annoy
from captum.attr import GradientShap
import matplotlib.pyplot as plt
import pandas as pd

def progenitor_tfs_detection(model,ay, countsInVitroCscMatrix, cellcell_score,metadata, clone_data, xp, pca_all, n_tfs = 100):
    set_seed()
    device = torch.device('cuda')
    NS = 100
    num_steps = int(4/0.1)

    day2 = get_day2_progenitors(metadata, clone_data)
    ay = annoy.AnnoyIndex(50,'euclidean')
    ay.load('/content/drive/MyDrive/SrJ/Weinreb/50_20_10.ann')
    with open('/content/drive/MyDrive/SrJ/Weinreb/50_20_10.txt', 'r') as f:
        cy = np.array([line.strip() for line in f])

    x_i = torch.tensor(xp[day2], dtype=torch.float32).to(device)
    c_i = torch.tensor(np.array(cellcell_score[day2])).float().to(device)
    c_i = c_i.requires_grad_()
    x_i = x_i.requires_grad_()
    monocytes = []
    neutrophils = []
    others = []
    x_i, c_i = model.forward(x_i, c_i)
    x_i = x_i.detach().cpu().numpy()
    for i in tqdm(range(len(day2))):
        nnn = cy[ay.get_nns_by_vector(x_i[i], 20)]
        nnn = Counter(nnn).most_common(2)
        label, num = nnn[0]
        if len(nnn) > 1:
            _, num2 = nnn[1]
            if num == num2:  # deal with ties by setting it to the default class
                label = 'Other'
        if label == "Monocyte":
            monocytes.append(i)
        elif label == "Neutrophil":
            neutrophils.append(i)
        else:
            others.append(i)

    print("Neutrophil & Monocyte Index Determined...")

    wrapper = WrapperModelx(model,pca_all).to(device)
    gs = GradientShap(wrapper)

    baselines = countsInVitroCscMatrix.iloc[day2].values.mean(0).reshape(1,-1)

    #Calculate the attribution scores for monocyte
    x_i = torch.tensor(countsInVitroCscMatrix.iloc[np.array(day2)[monocytes]].values, dtype=torch.float32).to(device)
    c_i = torch.tensor(np.array(cellcell_score[np.array(day2)[monocytes]])).float().to(device)
    c_i = c_i.requires_grad_()
    x_i = x_i.requires_grad_()
    attr_mono = gs.attribute(x_i,baselines = torch.tensor(baselines).float().to(device),
                    additional_forward_args =c_i)

    #Calculate the attribution scores for neutrophil
    x_i = torch.tensor(countsInVitroCscMatrix.iloc[np.array(day2)[neutrophils]].values, dtype=torch.float32).to(device)
    c_i = torch.tensor(np.array(cellcell_score[np.array(day2)[neutrophils]])).float().to(device)
    c_i = c_i.requires_grad_()
    x_i = x_i.requires_grad_()
    attr_neutro = gs.attribute(x_i,baselines = torch.tensor(baselines).float().to(device),
                    additional_forward_args =c_i)


    #all tfs of progenitors
    monocyte_progenitor_tfs  = ['F13a1', 'Ms4a6c','Ly6c2','S100a4','Rassf4','Csf1r','Hpse',
                                'Ly86','Emb','Papss2','Ctss','Slpi','Irf8','Nr4a1','Klf4']
                                    #turn on neutrophil progenitors tfs
    neutrophil_progenitor_tfs = ['Ltf','Ngp','Lcn2','Cd177','Camp','S100a9','Ifitm6','Itgb2l','Pglyrp1','S100a8',
                               'Lrg1','Fcnb','Gp1bb','Lyz2','Syne1']

    tfs ={}
    #Plot the attribution scores for monocyte
    contrib = attr_mono.mean(0).detach().cpu().numpy()
    genes = countsInVitroCscMatrix.columns
    scores = [(x,y) for x, y in zip(genes,contrib)]
    sorted_scores = list(reversed(sorted(scores, key = lambda x : abs(x[1]))))
    #Progenitor Tfs in Top 30
    print("Align with Monocyte Tfs ",len(set([x[0] for x in sorted_scores[:n_tfs]]).intersection(monocyte_progenitor_tfs)))


    df = pd.DataFrame(sorted_scores[:n_tfs], columns = ['Gene','Score'])
    df = df.reset_index()
    df.index = df.Gene
    fig, ax = plt.subplots(figsize=(24, 4))
    y_min = np.min(abs(df["Score"]))
    y_max = np.max(abs(df["Score"]))

    y_min -= 0.1 * (y_max - y_min)
    y_max += 0.4 * (y_max - y_min)
    ax.set_ylim(y_min, y_max)

    ax.set_xlim(-0.75, 100.5)

    for gene in df.index:
        if gene in monocyte_progenitor_tfs:
            color = "#FEAE00"
        else:
            color = "#000000"
        ax.text(
            df.loc[gene, "index"],
            abs(df.loc[gene, "Score"]),
            gene,
            rotation="vertical",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=16,
            color=color,
        )
    plt.show()

    tfs['mono'] = df['Gene'].values[:n_tfs]

    # Plot For Neutrophil
    contrib = attr_neutro.mean(0).detach().cpu().numpy()
    scores = [(x,y) for x, y in zip(genes,contrib)]
    sorted_scores = list(reversed(sorted(scores, key = lambda x : abs(x[1]))))
    print("Align with Neutrophil Tfs ",len(set([x[0] for x in sorted_scores[:n_tfs]]).intersection(neutrophil_progenitor_tfs)))

    df = pd.DataFrame(sorted_scores[:n_tfs], columns = ['Gene','Score'])
    df = df.reset_index()
    df.index = df.Gene
    fig, ax = plt.subplots(figsize=(24, 4))
    y_min = np.min(abs(df["Score"]))
    y_max = np.max(abs(df["Score"]))

    y_min -= 0.1 * (y_max - y_min)
    y_max += 0.4 * (y_max - y_min)
    ax.set_ylim(y_min, y_max)

    ax.set_xlim(-0.75, 100.5)

    for gene in df.index:
        if gene in neutrophil_progenitor_tfs:
            color = "#FEAE00"
        else:
            color = "#000000"
        ax.text(
            df.loc[gene, "index"],
            abs(df.loc[gene, "Score"]),
            gene,
            rotation="vertical",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=16,
            color=color,
        )
    plt.show()
    tfs['neutro'] = df['Gene'].values[:n_tfs]

    return tfs

        



