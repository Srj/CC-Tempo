import torch
from sklearn import preprocessing
import sklearn
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

class CellClassifier(torch.nn.Module):

    def __init__(self, gene_dim):
        super(CellClassifier, self).__init__()

        self.gene_dim = gene_dim
        self.fc1 = torch.nn.Linear(self.gene_dim, 100)
        self.fc2 = torch.nn.Linear(100, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


#define train loader
def train_loader_fn(gene,cc, label):
    device = torch.device('cuda')
    gene = torch.from_numpy(gene).float()
    cc = torch.from_numpy(cc).float()
    #concat input
    expr = torch.cat((gene,cc),1)

    label = torch.from_numpy(label)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(expr, label),
        batch_size=64, shuffle=True)
    return train_loader


#define test loader
def test_loader_fn(gene,cc, label):
    device = torch.device('cuda')
    gene = torch.from_numpy(gene).float()
    cc = torch.from_numpy(cc).float()
        #concat input
    expr = torch.cat((gene,cc),1)

    label = torch.from_numpy(label)

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(expr, label),
        batch_size=128, shuffle=False)
    return test_loader

#train the model
def train(model, train_loader, test_loader, optimizer, epoch):
    device = torch.device('cuda')
    for e in range(epoch):
        model.train()
        for batch_idx, (expr, label) in enumerate(train_loader):
            expr = expr.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(expr)
            loss = torch.nn.functional.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e, batch_idx * len(expr), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        #test the model
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for expr, label in test_loader:
                expr = expr.to(device)
                label = label.to(device)
                output = model(expr)
                test_loss += torch.nn.functional.cross_entropy(output, label, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()
            
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))

def main():
    #Read All the necessary files
    countsInVitroCscMatrix = pd.read_csv("/content/drive/MyDrive/SrJ/Weinreb/data/all/full_dataset_normalized_scaled.csv")
    # cellcellScore = pd.read_csv("/content/drive/MyDrive/SrJ/Weinreb/data/all/cell_cell_interaction_score.csv")
    metadata = pd.read_csv("/content/drive/MyDrive/SrJ/Weinreb/data/all/full_dataset_metadata.csv")

    #process
    pca_all =  PCA(n_components = 50, random_state = 47)
    # um = umap.UMAP(n_components = 2, metric = 'euclidean', n_neighbors = 30)
    xp = pca_all.fit_transform(countsInVitroCscMatrix)
    cellcell_score = pd.read_csv("/content/drive/MyDrive/SrJ/Weinreb/R/cellcellscore/cell_cell_interaction_score_Monocyte_0_Neutrophil_0.csv").values


    #define model
    device = torch.device('cuda')
    model = CellClassifier(gene_dim=50 + 58).to(device)
    #define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    yc = metadata['Annotation']
    #replace the value where the condition is true
    yc = yc.mask(~((yc == 'Neutrophil') | (yc == 'Monocyte')), 'Other')
    yc.value_counts()

    le = preprocessing.LabelEncoder()
    yc = le.fit_transform(yc)
    xtr, xte, ctr, cte, ytr, yte = sklearn.model_selection.train_test_split(xp ,cellcell_score, yc, stratify = yc)
    print(xtr.shape, xte.shape, ctr.shape, cte.shape, ytr.shape, yte.shape)

    #define model
    device = torch.device('cuda')
    model = CellClassifier(gene_dim=50 + 58).to(device)
    #define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = train_loader_fn(xtr,ctr,ytr)
    test_loader = test_loader_fn(xte,cte,yte)
    train(model, train_loader, test_loader, optimizer, 5)

    #test the model
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = []
    with torch.no_grad():
        for expr, label in test_loader:
            expr = expr.to(device)
            label = label.to(device)
            output = model(expr)
            pred = output.max(1, keepdim=True)[1].detach().cpu().numpy()
            y_pred.append(pred)

    y_pred = np.concatenate(y_pred).ravel().astype(int)
    print(sklearn.metrics.classification_report(yte, y_pred))
    print(sklearn.metrics.f1_score(yte, y_pred, average = 'macro'))
    torch.save(model.state_dict(), "/content/drive/MyDrive/SrJ/Weinreb/Classifier/classifier.pt")