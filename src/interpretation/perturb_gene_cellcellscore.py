import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import itertools
from scipy.io import mmread
from ligand_receptor_dict import *

sc = StandardScaler()
df = mmread("/content/drive/MyDrive/SrJ/Weinreb/data/raw/stateFate_inVitro_normed_counts.mtx")
gene_names = pd.read_csv("/content/drive/MyDrive/SrJ/Weinreb/data/raw/stateFate_inVitro_gene_names.txt",header = None)
outgoing_comm = pd.read_csv(f"/content/drive/MyDrive/SrJ/Weinreb/R/outgoing_comm_weinreb_mat_2.csv",index_col=0)
metadata = pd.read_csv("/content/drive/MyDrive/SrJ/Weinreb/data/raw/stateFate_inVitro_metadata.txt",sep='\t')
celltypes = metadata['Cell type annotation'].tolist()
cell_index_by_type = {celltype:[]for celltype in outgoing_comm.index.tolist()}

for i,c in enumerate(celltypes):
  cell_index_by_type[c].append(i)

#progenitor tfs needs to be updated
monocyte = ['Csf1r']
neutrophil = ['Itgb2l','Gp1bb']
TS = 'Time point'

for m, n in itertools.product([0,1,2,2.5],[0,-1,-2,-2.5]):
  gc.collect()
  df_copy = df.todense()
  df_copy = sc.fit_transform(np.array(df_copy))
  df_copy = pd.DataFrame(df_copy, columns = gene_names[0])
  gc.collect()
  print("m:",m,"n:",n)
  #recalculate ligand receptor scores
  #perturb tfs values
  if m != 0:
    df_copy[monocyte] = m
  if n != 0:
    df_copy[neutrophil] = n

  #score ligands
  ligand_score = pd.DataFrame(index = df_copy.index, columns = ligand_scores.keys())
  for pathway, genes in ligand_scores.items():
    ligand_score[pathway] = df_copy[genes].sum(axis = 1) / np.sqrt(len(genes))
  # ligand_score.to_csv("ligand_scoring_29_python_zscore.csv", index= False)

  #score receptors
  receptor_score = pd.DataFrame(index = df_copy.index, columns = receptor_scores.keys())
  for pathway, genes in receptor_scores.items():
    if 'Il6r' in genes:
      genes.remove('Il6r')
    receptor_score[pathway] = df_copy[genes].sum(axis = 1) / np.sqrt(len(genes))
  # receptor_score.to_csv("receptor_scoring_29_python_zscore.csv", index= False)
  print("Ligand Receptor Scoring Done")

  del df_copy

  for DAY in [2,4,6]:
      fname = f"cell_cell_interaction_score_{DAY}_Monocyte_{m}_Neutrophil_{n}.csv"
      ligand_score.rename({'ITGAL_ITGB2':'ITGAL-ITGB2'},axis = 1, inplace = True)
      receptor_score.rename({'ITGAL_ITGB2':'ITGAL-ITGB2'},axis = 1, inplace = True)
      incoming_comm = pd.read_csv(f"/content/drive/MyDrive/SrJ/Weinreb/R/incoming_comm_weinreb_mat_{DAY}.csv",index_col=0)
      outgoing_comm = pd.read_csv(f"/content/drive/MyDrive/SrJ/Weinreb/R/outgoing_comm_weinreb_mat_{DAY}.csv",index_col=0)
      extra_pathways = list(set(ligand_score.columns).difference(set(incoming_comm.columns)))
      incoming_comm.loc[:,extra_pathways] = 0
      extra_pathways = list(set(receptor_score.columns).difference(set(outgoing_comm.columns)))
      outgoing_comm.loc[:,extra_pathways] = 0
      assert incoming_comm.shape[1] == 29
      assert outgoing_comm.shape[1] == 29

      #Order Columns in the same way
      ligand_score = ligand_score.reindex(sorted(ligand_score.columns),axis=1)
      receptor_score = receptor_score.reindex(sorted(receptor_score.columns),axis=1)
      incoming_comm  = incoming_comm .reindex(sorted(incoming_comm .columns),axis=1)
      outgoing_comm = outgoing_comm.reindex(sorted(outgoing_comm.columns),axis=1)

      # outgoing x ligand
      sending_score = pd.DataFrame(index = ligand_score.index, columns = ligand_score.columns)

      outgoing_comm['CellID'] = cell_index_by_type
      exploded_mat = outgoing_comm.explode('CellID')
      exploded_mat.index = exploded_mat.CellID
      exploded_mat = exploded_mat.sort_index()
      exploded_mat = exploded_mat.drop('CellID',axis=1)

      sending_score = exploded_mat * ligand_score
      print(f"Calculating Sending Score for m: {m}, n: {n}, Day: {DAY}")

      # incoming x receptor
      receiving_score = pd.DataFrame(index = receptor_score.index, columns = receptor_score.columns)

      incoming_comm['CellID'] = cell_index_by_type
      exploded_mat = incoming_comm.explode('CellID')
      exploded_mat.index = exploded_mat.CellID
      exploded_mat = exploded_mat.sort_index()
      exploded_mat = exploded_mat.drop('CellID',axis=1)

      receiving_score = exploded_mat * receptor_score
      print(f"Calculating Receiving Score for m: {m}, n: {n}, Day: {DAY}")

      pd.concat([sending_score,receiving_score],axis = 1).to_csv(fname, index = False)
      print(f"Saved {fname}")

  #Combine All Day Data
  cellcell_score_2 = pd.read_csv(f"cell_cell_interaction_score_{2}_Monocyte_{int(m)}_Neutrophil_{int(n)}.csv")
  cellcell_score_4 = pd.read_csv(f"cell_cell_interaction_score_{4}_Monocyte_{int(m)}_Neutrophil_{int(n)}.csv")
  cellcell_score_6 = pd.read_csv(f"cell_cell_interaction_score_{6}_Monocyte_{int(m)}_Neutrophil_{int(n)}.csv")

  cellcell_score = pd.DataFrame(index = cellcell_score_2.index, columns = cellcell_score_2.columns)
  cellcell_score[metadata[TS] == 2] = cellcell_score_2[metadata[TS] == 2]
  cellcell_score[metadata[TS] == 4] = cellcell_score_4[metadata[TS] == 4]
  cellcell_score[metadata[TS] == 6] = cellcell_score_6[metadata[TS] == 6]
  cellcell_score.to_csv(f"/content/drive/MyDrive/SrJ/Weinreb/R/cellcellscore/cell_cell_interaction_score_Monocyte_{m}_Neutrophil_{n}.csv", index = False)
  print(f"Saved /content/drive/MyDrive/SrJ/Weinreb/R/cellcellscore/cell_cell_interaction_score_Monocyte_{m}_Neutrophil_{n}.csv")