import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

df = pd.read_csv("./data/procesado.csv")
df = df.drop(columns=["Unnamed: 0"])
df = df.drop(columns=["fieldPosition", "nonpossessionTeamId"])

def train(X_train, X_test, y_train, y_test, model, loss_fn, optimizer):
    n_epochs = 230
    batch_size = 2250
    batches_per_epoch = len(X_train) // batch_size

    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        # set model in training mode and run through each batch
        model.train()
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch.argmax(dim=1))
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # compute and store metrics
                acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(X_test)
        ce = loss_fn(y_pred, y_test)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)
        print(acc)
        train_loss.append(np.mean(epoch_loss))
        train_acc.append(np.mean(epoch_acc))
        test_loss.append(ce)
        test_acc.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch} validation: Cross-entropy={ce}, Accuracy={acc}")
        #print(f"Epoch {epoch} validation: Accuracy={acc}")
        
    model.load_state_dict(best_weights)
    graficar(train_loss, test_loss, train_acc, test_acc)
    y_pred=np.argmax(y_pred.detach().numpy(), axis=1)
    y_test=np.argmax(y_test.detach().numpy(), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    np.savetxt("../output/confusion.txt",cm,delimiter="|",newline="\n")

def graficar(train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist):
    plt.plot(train_loss_hist, label="train")
    plt.plot(test_loss_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.savefig('../output/entropy.png')
    plt.show()
    
    plt.plot(train_acc_hist, label="train")
    plt.plot(test_acc_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig('../output/accuracy.png')
    plt.show()

def preparar(data):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(data[['playTypeDetailed']])
    #print(ohe.categories_)
    
    y = ohe.transform(data[['playTypeDetailed']])
    data = data.drop(columns=["playTypeDetailed"])
    data.values.astype(np.float32)
    #print(y)
    X = torch.tensor(data.values, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(8, 1620)
        self.act = nn.ReLU()
        self.output = nn.Linear(1620, 40)
        
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x
        
X, y = preparar(df)

X_t, X_tst, y_t, y_tst = train_test_split(X, y, train_size=0.7, shuffle=True)

modl = Multiclass()

loss = nn.CrossEntropyLoss()
opt = optim.Adam(modl.parameters(), lr=0.001)

train(X_t, X_tst, y_t, y_tst, modl, loss, opt)
