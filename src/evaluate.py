import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix
import os
import torch
import sklearn
# from model import AutoGenerator
from prescient.train import AutoGenerator
import glob
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from types import SimpleNamespace
from scipy.stats import pearsonr


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--cc_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='experiments/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=int, required=True)
    return parser

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

def load_model(obj, path, epoch="train.best.pt", device='cuda'):
    config_path = os.path.join(path, "config.pt")
    config = SimpleNamespace(**torch.load(config_path))
    model = obj(config)
    train_pt = os.path.join(path, "train.best.pt")
    checkpoint = torch.load(train_pt, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model


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
    day2 = clone_metadata.index[(clone_metadata['Time point'] == 2.0) & (clone_metadata['Annotation'] == 'undiff')]
    print("Number of Progenitor Cells:",len(day2))
    return day2

def get_base_fate_bias(metadata, clone_data, day2):
    #get the cells from day 4 & 6
    day4_6 = metadata.loc[metadata['Time point'] > 3]
    day4_6_neutrophil = day4_6.loc[(day4_6['Annotation'] == 'Neutrophil')]
    day4_6_monocyte = day4_6.loc[(day4_6['Annotation'] == 'Monocyte')]
    #get day 4 & 6 Neu/Monocyte cell's indices
    neutrophil_indices = day4_6_neutrophil.index.tolist()
    monocyte_indices = day4_6_monocyte.index.tolist()

    #for each cell in test set
    base_fate_bias = []
    for i in range(len(day2)):
        #find the index of the clone this test cell belongs to
        clone_index = np.where(clone_data[day2[i]] == 1)[0]
        #get the indices of cells who belongs to this clone
        indices = np.where(clone_data[:, clone_index] == 1)[0].tolist()
        #get the indices of neutrophil or monocyte that belongs to this clone
        indices_n = list(set(indices) & set(neutrophil_indices))
        indices_m = list(set(indices) & set(monocyte_indices))

        #count the number of neutrophil and monocyte
        n_count = len(indices_n)
        m_count = len(indices_m)
        total = n_count + m_count
        base_fate_bias.append((n_count+1)/(total+2))
    return np.array(base_fate_bias)

def evaluate_clonal_fate_bias(model, xs,ccs, lr, NS=100, num_steps=int(4/0.1), device='cuda'):
    set_seed(42)
    scores = []
    cor_mask = []

    assert len(xs) == len(ccs), "xs and ccs must be the same length"

    for i in tqdm(range(len(xs))):
        #Expand Data Point
        x_i = torch.tensor(xs[i].reshape(1,-1)).float().expand(NS,-1).to(device)
        c_i = torch.tensor(ccs[i].reshape(1,-1)).float().expand(NS,-1).to(device)

        #Simulate Forward
        for _ in range(num_steps):
            z = torch.randn(x_i.shape[0], x_i.shape[1]) * 0.5
            z = z.to(device)
            x_i, c_i = model._step(x_i, dt=0.1, z=z, y=c_i)
        x_i = x_i.detach().cpu().numpy()

        #classify
        y_pred = lr.predict(x_i)
        # may want to save yp instead
        num_neu = sum(y_pred == 'Neutrophil') # use pseudocounts for scoring
        num_mono = sum(y_pred == 'Monocyte')
        num_total =  num_neu + num_mono
        # print(i,num_neu, num_mono)
        score = (num_neu + 1) / (num_total + 2)
        mask = (num_total > 0)
        scores.append(score)
        cor_mask.append(mask)

    scores = np.array(scores)
    cor_mask = np.array(cor_mask)

    return scores, cor_mask



def evaluate_model(model_dir, data_dir, cc_dir, device):
    #load the data
    # countsInVitroCscMatrix = pd.read_csv(os.path.join(data_dir,"full_dataset_normalized_scaled.csv"))
    metadata = pd.read_csv(os.path.join(data_dir,"full_dataset_metadata.csv"))

    clone_data = mmread(os.path.join(data_dir,'stateFate_inVitro_clone_matrix.mtx'))
    clone_data = csc_matrix(clone_data, shape=(130887, 5864)).toarray()

    XpXu = torch.load(os.path.join(data_dir,"XpXu.pt"))
    xp = XpXu['xp']

    cellcell_score = pd.read_csv(cc_dir)
    cellcell_score = cellcell_score.to_numpy().astype(float)


    #get the progenitors
    day2 = get_day2_progenitors(metadata, clone_data)

    #get the base fate bias
    base_fate_bias = get_base_fate_bias(metadata, clone_data, day2)

    #train the logistic regression model
    y = metadata['Annotation'].to_numpy()
    X = xp

    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)

    print("Data Prepared and Loaded")
    print("Loading Model")

    #load models
    model_files = glob.glob(os.path.join(model_dir,"*"))   #CC-Tempo
    # model_files = glob.glob(f"{model_dir}/*/*")
    print(f"Evaluating {len(model_files)} Models")


    clonal_fate_bias_scores = pd.DataFrame(columns=["Model", "PCC", "PCC_Masked"])
    for i, model in enumerate(model_files):
        print(f"Evaluating {model}")
        model = load_model(AutoGenerator, model).to(device)
        scores, mask = evaluate_clonal_fate_bias(model, xp[np.array(day2)], cellcell_score[np.array(day2)], lr, device=device)
        r, pval = pearsonr(base_fate_bias, scores)
        print(model, r)
        r_masked, pval = pearsonr(np.array(base_fate_bias)[mask], scores[mask])
        print(r_masked)
        clonal_fate_bias_scores = clonal_fate_bias_scores._append({"Model": i, "PCC": r, "PCC_Masked": r_masked}, ignore_index=True)

    return clonal_fate_bias_scores

def main():
    parser = create_parser()
    args = parser.parse_args()
    print("Path:")
    print(args.data_path)
    print(args.model_path)
    print(args.cc_path)
    print(args.device)
    df = evaluate_model(args.model_path, args.data_path, args.cc_path, args.device)
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()


"""
python src/evaluate.py --data_path data/ --cc_path data/cc_score_random.csv 
--model_path experiments_random_noise/ --output fate_bias_random_score.csv
"""





