import torch
import torch.nn as nn
import pandas as pd
import copy
import os

from data import ListingsDataset
from model import RiskModel


hidden_size1 = 128 
hidden_size2 = 64
hidden_size3 = 16
ouput_size = 2
num_epochs = 50
batch_size = 8
learning_rate = 0.003  # Learning rate for the optimizer

train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'mega_training.csv')
val_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'short_val.csv')

train_dataset = ListingsDataset(dataset_path=train_path, dataset_type="train")
validation_dataset = ListingsDataset(dataset_path=val_path, dataset_type="train")

input_size = train_dataset.input_dimension # 22

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
print("All data loaded.")

model = RiskModel(input_size=input_size, hidden1=hidden_size1, hidden2=hidden_size2, hidden3=hidden_size3, output=ouput_size)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

min_loss = 0
min_epoch = 0


for epoch in range(num_epochs):
    model.train()
    for i, (data, labels) in enumerate(train_loader):

        outputs = model(data)
        _labels = torch.cat([1. - labels, labels], dim=1)
        loss = criterion(outputs, _labels)


        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10000 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
    
    # torch.save(model.state_dict, f'epochs\epoch{epoch}.pth')
    
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    for i, (data, labels) in enumerate(validation_loader):

        outputs = model(data)
        _labels = torch.cat([1. - labels, labels], dim=1)
        loss = criterion(outputs, _labels)
        total_loss += loss
        rounded_outputs = torch.round(outputs)

        correct_preds += (rounded_outputs == _labels).sum().item()
        total_preds += _labels.shape(0)

    if min_loss == 0:
        min_loss = total_loss
        min_epoch = epoch
    elif total_loss < min_loss:
        min_loss = total_loss
        min_epoch = epoch

    accuracy = correct_preds / total_preds
    print(f'Accuracy: {accuracy * 100:.2f}% at epoch: {epoch}')
    print(f'epoch {epoch}, loss = {total_loss:4f}, min loss = {min_loss:4f} at epoch: {min_epoch}')
    #write output and expected output to a csv file
        

print(f'min loss:  {min_loss} at epoch: {min_epoch}')


# torch.save(model.state_dict(), 'epochs\epoch#1100.pth') 
# model.load_state_dict(torch.load('epochs\epoch#254.pth'))

predictions = []
rounded_preds = []
expected_vals = []
with torch.no_grad():
    for i, (data, labels) in enumerate(validation_loader):
        outputs = model(data)
        predictions.append(outputs)
        rounded_preds.append(torch.round(outputs))
        expected_vals.append(labels)

predictions = torch.cat(predictions, dim=0)
rounded_preds = torch.cat(rounded_preds, dim=0)
expected_vals = torch.cat(expected_vals, dim=0)

predictions = predictions.ravel()
rounded_preds = rounded_preds.ravel()
expected_vals = expected_vals.ravel()

results_df = pd.DataFrame({'Prediction' : predictions, 'Rounded prediction' : rounded_preds, 'Expected value' : expected_vals})

results_df.to_csv('risk_percentages1.csv', index=False)


'''
model.eval()
correct_preds = 0
total_preds = 0

with torch.no_grad():
    for data, labels in validation_loader:
        outputs = model(data)
        # Round the predictions to the nearest integer (or use your own rounding logic)
        rounded_outputs = torch.round(outputs)

        correct_preds += (rounded_outputs == labels).sum().item()
        total_preds += labels.size(0)

accuracy = correct_preds / total_preds
print(f'Accuracy: {accuracy * 100:.2f}%')
'''