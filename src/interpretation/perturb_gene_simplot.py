import matplotlib.pyplot as plt
import seaborn as sns
import torch

def perturb_gene_plot(count_dict_m = None, count_dict_n = None):
  if count_dict_m is None:
    count_dict = torch.load("/content/drive/MyDrive/SrJ/Weinreb/Interpretation/perterbation_simulation_data_monocyte_up_100sim.pt")["count"]
  else:
    count_dict = count_dict_m
  fig, ax = plt.subplots(4,4, sharex = True, sharey = True,figsize=(8,8))
  for j,m in enumerate([0,1,2,2.5]):
    for i,n in enumerate([0,-1,-2,-2.5]):
      sns.barplot(data = count_dict[m,n]/len(count_dict[m,n]) * 100, ax= ax[i,j])
      if i == 0:
        ax[i,j].set_xlabel(m, rotation = 90)
      if j == 0:
        ax[i,j].set_ylabel(n)
      ax[i,j].xaxis.set_label_position('top')
      ax[i,j].set_xticklabels(ax[i,j].get_xticklabels(), rotation=90, ha='right')
  fig.supxlabel("Monocyte",y = 0.95)
  fig.supylabel("Neutrophil")
  plt.show()

  if count_dict_n is None:
    count_dict = torch.load("/content/drive/MyDrive/SrJ/Weinreb/Interpretation/perterbation_simulation_data_neutrophil_up_100sim.pt")["count"]
  else:
    count_dict = count_dict_n
  fig, ax = plt.subplots(4,4, sharex = True, sharey = True,figsize=(8,8))
  for i,n in enumerate([0,1,2,2.5]):
    for j,m in enumerate([0,-1,-2,-2.5]):
      sns.barplot(data = count_dict[m,n]/len(count_dict[m,n]) * 100, ax= ax[j,i])
      if j == 0:
        ax[j,i].set_xlabel(n, rotation = 90)
      if i == 0:
        ax[j,i].set_ylabel(m)
      ax[j,i].xaxis.set_label_position('top')
      ax[j,i].set_xticklabels(ax[j,i].get_xticklabels(), rotation=90, ha='right')
  fig.supylabel("Monocyte",)
  fig.supxlabel("Neutrophil",y = 0.95)
  plt.show()