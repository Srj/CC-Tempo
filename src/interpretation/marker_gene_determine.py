from utils import set_seed, get_day2_progenitors
from models import InferenceAutoGenerator
import torch
from types import SimpleNamespace
import annoy
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
from captum.attr import KernelShap
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda')
NS = 100
num_steps = int(4/0.1)
config_path = "/content/drive/MyDrive/SrJ/Weinreb/NewCellChat_108_2Branch/NoRandom/experiments/kegg-growth-softplus_1_500-1e-06/seed_2/config.pt"
config = SimpleNamespace(**torch.load(config_path))
model = InferenceAutoGenerator(config)
train_pt = "/content/drive/MyDrive/SrJ/Weinreb/NewCellChat_108_2Branch/NoRandom/experiments/kegg-growth-softplus_1_500-1e-06/seed_2/train.best.pt"
checkpoint = torch.load(train_pt, map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

ay = annoy.AnnoyIndex(50,'euclidean')
ay.load('/content/drive/MyDrive/SrJ/Weinreb/50_20_10.ann')
with open('/content/drive/MyDrive/SrJ/Weinreb/50_20_10.txt', 'r') as f:
    cy = np.array([line.strip() for line in f])



def marker_detection(model,ay, countsInVitroCscMatrix, cellcell_score,metadata, clone_data, xp, pca_all):
    set_seed()

    day2 = get_day2_progenitors(metadata, clone_data)
    day6 = metadata.loc[metadata['Time point'] > 5].index.tolist()
    
    x_i = torch.tensor(xp[day2], dtype=torch.float32).to(device)
    c_i = torch.tensor(np.array(cellcell_score[day2])).float().to(device)
    c_i = c_i.requires_grad_()
    x_i = x_i.requires_grad_()


    day2 = get_day2_progenitors(metadata, clone_data)
    day6 = metadata.loc[metadata['Time point'] > 5].index.tolist()

    monocytes = []
    neutrophils = []
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

    print("Neutrophil & Monocyte Index Determined...")

    #set baseline value for all genes
    baselines = countsInVitroCscMatrix.iloc[day6].values.mean(0)

    #convert back x_i to gene space
    x_i = pca_all.inverse_transform(x_i)

    def potential(x_i,c_i):
        x_i = x_i.detach().cpu().numpy()
        x_i = pca_all.transform(x_i)
        x_i = torch.tensor(x_i).to(device).float()
        return model._pot(x_i,c_i)

    #Define Kernel Shap
    ks = KernelShap(potential)

    #Monocyte
    attr_mono = ks.attribute(torch.tensor(x_i[monocytes]).float().to(device),baselines = torch.tensor(baselines).float().to(device),
                additional_forward_args =c_i[monocytes].float().to(device),show_progress = False)

    attr_neutro = ks.attribute(torch.tensor(x_i[neutrophils]).float().to(device),baselines = torch.tensor(baselines).float().to(device),
                        additional_forward_args =c_i[neutrophils].float().to(device),show_progress = False)

    print("Shap Score Determined...\nEvaluating Marker Gene...")

    #Using Seurat
    markers = pd.read_csv("/content/drive/MyDrive/SrJ/Weinreb/R/weinreb_markers_by_cluster.csv")
    #From Weinreb Paper
    monocyte_markers = ['Ms4a6d', 'Fabp5', 'Ctss', 'Ms4a6c', 'Tgfbi', 'Olfm1', 'Csf1r', 'Ccr2', 'Klf4', 'F13a1']
    neutrophil_markers = ['S100a9','Itgb2l','Elane','Fcnb','Mpo','Prtn3','S100a6','S100a8','Lcn2','Lrg1']
    genes = countsInVitroCscMatrix.columns


    contrib = attr_mono.mean(0).detach().cpu().numpy()
    scores = [(x,y) for x, y in zip(genes,contrib)]
    sorted_scores = list(reversed(sorted(scores, key = lambda x : abs(x[1]))))
    print("Monocyte")
    #Seurat Markers in Top 30
    print("Align with Seurat ",len(set([x[0] for x in sorted_scores[:30]]).intersection(markers[markers['cluster'] =='Monocyte'].gene)))
    #Seurat Markers in Top 30
    print("Align with Weinreb ",len(set([x[0] for x in sorted_scores[:30]]).intersection(monocyte_markers)))

    print("--------------------------------------------")
    print("Neutrophil")
    contrib = attr_neutro.mean(0).detach().cpu().numpy()
    scores = [(x,y) for x, y in zip(genes,contrib)]
    sorted_scores = list(reversed(sorted(scores, key = lambda x : abs(x[1]))))
    #Seurat Markers in Top 30
    print("Align with Seurat ",len(set([x[0] for x in sorted_scores[:30]]).intersection(markers[markers['cluster'] =='Neutrophil'].gene)))
    #Seurat Markers in Top 30
    print("Align with Weinreb ",len(set([x[0] for x in sorted_scores[:30]]).intersection(neutrophil_markers)))

    torch.save({'attr_mono': attr_mono, 'attr_neutro' : attr_neutro}, "/content/drive/MyDrive/SrJ/Weinreb/Interpretation/Figure3_Marker_Detection/shap_values.pt")
    # torch.save({'attr_mono': attr_mono, 'attr_neutro' : attr_neutro}, "Figure3_Marker_Detection/shap_values.pt")
    print("Score Saved... ")

def plot_markers(attr, genes, type = "Monocyte", n_markers=30):
    contrib = attr.mean(0).detach().cpu().numpy()
    scores = [(x,y) for x, y in zip(genes,contrib)]
    sorted_scores = list(reversed(sorted(scores, key = lambda x : abs(x[1]))))

    #Using Seurat
    markers = pd.read_csv("/content/drive/MyDrive/SrJ/Weinreb/R/weinreb_markers_by_cluster.csv")
    #From Weinreb Paper
    monocyte_markers = ['Ms4a6d', 'Fabp5', 'Ctss', 'Ms4a6c', 'Tgfbi', 'Olfm1', 'Csf1r', 'Ccr2', 'Klf4', 'F13a1']
    neutrophil_markers = ['S100a9','Itgb2l','Elane','Fcnb','Mpo','Prtn3','S100a6','S100a8','Lcn2','Lrg1']

    #Seurat Markers in Top 30
    print("Align with Seurat ",len(set([x[0] for x in sorted_scores[:n_markers]]).intersection(markers[markers['cluster'] ==type].gene)))
    #Seurat Markers in Top 30
    if type == "Monocyte":
        print("Align with Weinreb ",len(set([x[0] for x in sorted_scores[:n_markers]]).intersection(monocyte_markers)))
    else:
        print("Align with Weinreb ",len(set([x[0] for x in sorted_scores[:n_markers]]).intersection(neutrophil_markers)))

    df = pd.DataFrame(sorted_scores[:n_markers], columns = ['Gene','Score'])
    df = df.reset_index()
    df.index = df.Gene
    fig, ax = plt.subplots(figsize=(10, 4))
    y_min = np.min(abs(df["Score"]))
    y_max = np.max(abs(df["Score"]))

    y_min -= 0.1 * (y_max - y_min)
    y_max += 2
    ax.set_ylim(y_min, y_max)

    ax.set_xlim(-0.75, n_markers - 0.25)

    for gene in df.index:
        if gene in (neutrophil_markers if type == "Neutrophil" else monocyte_markers):
            color = "#FEAE00"
        elif gene in markers[markers['cluster'] ==type].gene.values:
            color = "#00AB8E"
        else:
            color = "#000000"
        ax.text(
            df.loc[gene, "index"],
            abs(df.loc[gene, "Score"]),
            gene,
            rotation="vertical",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=18,
            color=color,
        )
    plt.title(type.capitalize(), fontsize=20)
    plt.tight_layout()
    # plt.savefig(f"/content/drive/MyDrive/SrJ/Weinreb/Interpretation/Figure3_Marker_Detection/{type}_markers.pdf",dpi=300)
    plt.savefig(f"{type}_markers.pdf",dpi=300)
    return df['Gene'].values[:n_markers]

