import os
import pandas as pd

def process_cellcell_score(data_dir,types = ["incoming", "outgoing"]):
    for typ in types:
        df1 = pd.read_csv(os.path.join(data_dir, f"{typ}_comm_weinreb2.csv"))
        df2 = pd.read_csv(os.path.join(data_dir, f"{typ}_comm_weinreb4.csv"))
        df3 = pd.read_csv(os.path.join(data_dir, f"{typ}_comm_weinreb6.csv"))

        df1 = df1.dropna(axis = 0).pivot(columns ="Signaling", index="CellGroup", values="Contribution").fillna(0)
        df2 = df2.dropna(axis = 0).pivot(columns ="Signaling", index="CellGroup", values="Contribution").fillna(0)
        df3 = df3.dropna(axis = 0).pivot(columns ="Signaling", index="CellGroup", values="Contribution").fillna(0)

        output_pathways = set(df1.columns.tolist() + df2.columns.tolist() +  df3.columns.tolist())

        pathways = pd.read_csv(os.path.join(data_dir, "signalling_pathways.csv"))

        for DAY in [2,4,6]:
            ligand_scoring = pd.read_csv(f'./ligand_scoring_29_python_zscore.csv')
            receptor_scoring = pd.read_csv(f'./receptor_scoring_29_python_zscore.csv')
            ligand_scoring.rename({'ITGAL_ITGB2':'ITGAL-ITGB2'},axis = 1, inplace = True)
            receptor_scoring.rename({'ITGAL_ITGB2':'ITGAL-ITGB2'},axis = 1, inplace = True)
            incoming_comm = pd.read_csv(f"./incoming_comm_weinreb_mat_{DAY}.csv",index_col=0)
            outgoing_comm = pd.read_csv(f"./outgoing_comm_weinreb_mat_{DAY}.csv",index_col=0)
            extra_pathways = list(set(ligand_scoring.columns).difference(set(incoming_comm.columns)))
            extra_pathways = list(set(ligand_scoring.columns).difference(set(incoming_comm.columns)))
            incoming_comm.loc[:,extra_pathways] = 0
            outgoing_comm.loc[:,extra_pathways] = 0
            assert incoming_comm.shape[1] == 29
            assert outgoing_comm.shape[1] == 29
            
            #Order Columns in the same way
            ligand_scoring = ligand_scoring.reindex(sorted(ligand_scoring.columns),axis=1)
            receptor_scoring = receptor_scoring.reindex(sorted(receptor_scoring.columns),axis=1)
            incoming_comm  = incoming_comm .reindex(sorted(incoming_comm .columns),axis=1)
            outgoing_comm = outgoing_comm.reindex(sorted(outgoing_comm.columns),axis=1)
            
            # outgoing x ligand
            sending_score = pd.DataFrame(index = ligand_scoring.index, columns = ligand_scoring.columns)
            
            outgoing_comm['CellID'] = cell_index_by_type
            exploded_mat = outgoing_comm.explode('CellID')
            exploded_mat.index = exploded_mat.CellID
            exploded_mat = exploded_mat.sort_index()
            exploded_mat = exploded_mat.drop('CellID',axis=1)
            
            sending_score = exploded_mat * ligand_scoring
            print(f"Calculating Sending Score for m: {m}, n: {n}, Day: {DAY}")
                
            # incoming x receptor
            receiving_score = pd.DataFrame(index = receptor_scoring.index, columns = receptor_scoring.columns)
            
            incoming_comm['CellID'] = cell_index_by_type
            exploded_mat = incoming_comm.explode('CellID')
            exploded_mat.index = exploded_mat.CellID
            exploded_mat = exploded_mat.sort_index()
            exploded_mat = exploded_mat.drop('CellID',axis=1)
            
            receiving_score = exploded_mat * receptor_scoring
            print(f"Calculating Receiving Score for m: {m}, n: {n}, Day: {DAY}")
               
            pd.concat([sending_score,receiving_score],axis = 1).to_csv(fname, index = False)
            print(f"Saved {fname}")
            
        #Combine All Day Data
        cellcell_score_2 = pd.read_csv(f"cell_cell_interaction_score_{2}_Monocyte_{int(m)}Fold{M}_Neutrophil_{int(n)}Fold{N}.csv")
        cellcell_score_4 = pd.read_csv(f"cell_cell_interaction_score_{4}_Monocyte_{int(m)}Fold{M}_Neutrophil_{int(n)}Fold{N}.csv")
        cellcell_score_6 = pd.read_csv(f"cell_cell_interaction_score_{6}_Monocyte_{int(m)}Fold{M}_Neutrophil_{int(n)}Fold{N}.csv")

        cellcell_score = pd.DataFrame(index = cellcell_score_2.index, columns = cellcell_score_2.columns)
        cellcell_score[metadata[TS] == 2] = cellcell_score_2[metadata[TS] == 2]
        cellcell_score[metadata[TS] == 4] = cellcell_score_4[metadata[TS] == 4]
        cellcell_score[metadata[TS] == 6] = cellcell_score_6[metadata[TS] == 6]
        cellcell_score.to_csv(f"cellcellscore/cell_cell_interaction_score_Monocyte_{int(m)}Fold{M}_Neutrophil_{int(n)}Fold{N}.csv", index = False)
        print(f"Saved cellcellscore/cell_cell_interaction_score_Monocyte_{int(m)}Fold{M}_Neutrophil_{int(n)}Fold{N}.csv")
    
