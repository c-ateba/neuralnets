#Import Libraries
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split as ts
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn.functional as F

#Preprocess Data
torch.manual_seed(0) #to repeat results
data = pd.read_csv('iris.csv').set_index('Id')
dummies = pd.get_dummies(data['Species']) #from categoricals to dummies
data = pd.concat([data, dummies],axis=1).drop('Species',axis=1) #add dummies back
print(data.shape)

X_train, X_test, y_train, y_test = ts(data.iloc[:,:-3],data.iloc[:,-3:], test_size=0.2, shuffle=True, random_state=0 )
#convert data to tensors
X_train = torch.from_numpy(X_train.values).float()
X_test = torch.from_numpy(X_test.values).float()
y_train = torch.from_numpy(y_train.values).float()
y_test = torch.from_numpy(y_test.values).float()

#Create data loader
train_set = TensorDataset(X_train,y_train)
test_set = TensorDataset(X_test,y_test)
train_loader = DataLoader(train_set,batch_size=X_train.shape[0],shuffle=True,)
test_loader = DataLoader(test_set,batch_size=X_test.shape[0],shuffle=True,)

#Model parameters
learning_rate = 1e-3
input_size=4
output_size=3

#Create Pytorch model
class classifier(nn.Module):
    def __init__(self,input_size, output_size):
        super(classifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,16)
        self.fc4 = nn.Linear(16,output_size)

    def forward(self,input):
        x = torch.relu(self.fc1(input))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

classifier = classifier(input_size=input_size,output_size=output_size)
#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()
Optimizer = torch.optim.Adam(classifier.parameters(),lr=learning_rate)

def fit(Epochs, classifier, loss_fn, Optimizer, train_loader):
    losses = []
    epochs = []
    for epoch in range(Epochs):
        for xb, yb in train_loader:
            Optimizer.zero_grad()
            pred = classifier(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            Optimizer.step()
            losses.append(loss.item())
            epochs.append(epoch)
            if epoch%5==0:
                actuals = np.argmax(yb.detach(),axis=1)
                predicted = np.argmax(pred.detach(),axis=1)
                accuracy = accuracy_score(actuals, predicted)
                print('Epoch {}, accuracy is {}%, and loss is {}'.format(epoch,round(accuracy*100,0),loss.item()))
                print('The samples were:')
                print(actuals)
                print('predictions are:')
                print(predicted)
    plt.plot(epochs, losses)
    plt.xlabel('Training epoch')
    plt.ylabel('Model Loss')
    plt.show()

#Train the model
fit(100,classifier,loss_fn, Optimizer, train_loader)

#Test the model
for x,y in test_loader:
    preds = classifier(x)
    acts = np.argmax(y.detach(),axis=1)
    predicted = np.argmax(preds.detach(),axis=1)
    accuracy = accuracy_score(acts, predicted)
    print('Test Accuracy {}%'.format(round(accuracy*100,0)))
    print('The test samples were:')
    print(acts)
    print('The test predictions are:')
    print(predicted)


