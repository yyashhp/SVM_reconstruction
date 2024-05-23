import numpy as np
import torch
import pandas as pd
from svm_play_model import Binary_SVM

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#Get/Arrange Dataset

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data['data'], columns = data['feature_names'])
df['target'] = data['target']
#Target here is -1 or 1, benign or malignant

#Split datasets

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data['data'], 2 * data['target'] - 1, test_size = 0.2, random_state = seed)

#Load data/dataloaders
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Train
model = Binary_SVM(train_x.shape[1])
#Hyperparameters and Optimizer
epochs = 1000
C = 1.0
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_loss = 0
n_batch = 0

model.train()
for epoch in range(epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()

        train_x, train_y = batch
        output = model(train_x.float())
        loss = 0.5 * torch.norm(model.linear.weight.squeeze()) ** 2
        loss += C * torch.clamp(1 - train_y*output, min=0).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss
        n_batch += 1
    
    print(f"Epoch: {epoch}\t Loss:{total_loss / n_batch}")

#Evaluation

def accuracy(true_y, predicted_y):
    return (true_y == predicted_y).mean()

total_loss = 0
n_batch = 0
true_y = np.array([])
predicted_y = np.array([])

model.eval()
for batch in test_dataloader:
    test_x, test_y = batch
    output = model(test_x.float())
    loss = 0.5 * torch.norm(model.linear.weight.squeeze()) ** 2
    loss += C * torch.clamp(1 - test_y*output, min=0).mean()

    total_loss += loss
    n_batch += 1

    true_y = np.append(true_y, test_y.numpy())
    pred_y = 2 * (output >= 0) - 1
    predicted_y = np.append(predicted_y, pred_y.numpy())

print(f"Test Loss: {total_loss / n_batch} \t Accuracy: {accuracy(true_y, predicted_y)} ")