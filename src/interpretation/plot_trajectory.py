import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv
from holoviews import opts
hv.extension('matplotlib')
import pandas as pd
import numpy as np
from types import SimpleNamespace
import torch
import prescient

device = torch.device('cuda')

def plot_traj_holoviz(xu, model_out_list, NS = 100, name = 'test', return_plot = False):
  traj_df = pd.DataFrame(index = pd.MultiIndex.from_product((range(NS),np.arange(2,6.1,0.1)),names = ['Cell','Time']), columns = ('x','y'))
  for cell in range(NS):
    traj_df.loc[cell,:] = model_out_list[cell::NS]
  traj_df = traj_df.astype(np.float32)
  bg = hv.Scatter(xu).opts(color='grey', s= 0.5,aspect=0.8, fig_inches=20)
  traj_list = []
  for cell in range(NS):
    traj_list.append(hv.Path(traj_df.loc[cell].reset_index(), vdims='Time').opts(color='Time', linewidth=0.2, show_legend = True,
                                                                      colorbar = True,cmap = "coolwarm",aspect=0.8, fig_inches=20))
  pp = bg
  for i in traj_list:
    pp = pp * i
  if return_plot:
    return pp
  else:
    hv.save(pp,filename =f'{name}.png',dpi = 300)

def plot_perterbations(TARGET_CELL,M,N, mono_tfs, neutro_tfs, model, um, pca_all,countsInVitroCscMatrix, cellcell_score, clone_data,xu,prescient_plot = False):
    CLONE_ID = (clone_data[TARGET_CELL] == 1).nonzero()[0][0]
    NS = 100
    num_steps = int(4/0.1)
    if prescient_plot:
        #Prescient
        config_path = "/content/drive/MyDrive/SrJ/Weinreb/baseline_PC50/experiments/kegg-growth-softplus_1_500-1e-06/seed_2/config.pt"
        config = SimpleNamespace(**torch.load(config_path))
        model_p = prescient.train.AutoGenerator(config)
        train_pt = "/content/drive/MyDrive/SrJ/Weinreb/baseline_PC50/experiments/kegg-growth-softplus_1_500-1e-06/seed_2/train.best.pt"
        checkpoint = torch.load(train_pt, map_location = device)
        model_p.load_state_dict(checkpoint['model_state_dict'])
        model_p.to(device)
        # print(model)

        num_steps = int(4/0.1)

        x_i = countsInVitroCscMatrix.iloc[TARGET_CELL].copy()
        if M != 0:
            x_i[mono_tfs] = M
        if N != 0:
            x_i[neutro_tfs] = N

        x_i = pca_all.transform(x_i.values.reshape(1,-1))
        x_i = torch.tensor(x_i).float().expand(100,-1).to(device)

        p_outputs = [x_i.detach().cpu().numpy()]
        for _ in range(num_steps):
            z = torch.randn(x_i.shape[0], x_i.shape[1]) * 0.5
            z = z.to(device)
            x_i = model_p._step(x_i, dt = 0.1, z = z)
            p_outputs.append(x_i.detach().cpu().numpy())
        x_i = x_i.detach().cpu().numpy()
        p_out = um.transform(np.concatenate(p_outputs))

    
    x_i = countsInVitroCscMatrix.iloc[TARGET_CELL].copy()
    if M != 0:
        x_i[mono_tfs] = M
    if N != 0:
        x_i[neutro_tfs] = N

    x_i = pca_all.transform(x_i.values.reshape(1,-1))
    x_i = torch.tensor(x_i).float().expand(100,-1).to(device)
    c_i = torch.tensor(np.array(cellcell_score[TARGET_CELL]).reshape(1,-1)).float().expand(100,-1).to(device)

    m_outputs = [x_i.detach().cpu().numpy()]
    for _ in range(num_steps):
        z = torch.randn(x_i.shape[0], x_i.shape[1]) * 0.5
        z = z.to(device)
        x_i, c_i = model._step(x_i, dt = 0.1, z = z, y = c_i)
        m_outputs.append(x_i.detach().cpu().numpy())
    x_i = x_i.detach().cpu().numpy()
    model_out = um.transform(np.concatenate(m_outputs))

    p1 = plot_traj_holoviz(xu,model_out, NS = 100, name = 'test', return_plot = True)
    p1 = p1.opts(title = "Cell-Cell Interaction")

    if prescient_plot:
        p2 = plot_traj_holoviz(xu,p_out, NS = 100, name = 'test', return_plot = True)
        p2 = p2.opts(title = 'Prescient')
        p1 = p1 + p2

    hv.save(p1,filename =f'day2_Cell_{TARGET_CELL}_CLONE_{CLONE_ID}_MONOCYTE_{M}_NEUTROPHIL_{N}.png',dpi = 300)
    return p1