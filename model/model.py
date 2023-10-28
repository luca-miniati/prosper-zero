import torch
import torch.nn as nn

class RiskModel(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output, dropout_prob=0.5):
        super(RiskModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden1) 
        self.dropout1 = nn.Dropout(p=dropout_prob) 
        self.l2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(p=dropout_prob) 
        self.l3 = nn.Linear(hidden2, output)
        # self.l4 = nn.Linear(hidden3, hidden4)
        # self.l5 = nn.Linear(hidden4, output)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.dropout1(out) 
        out = self.relu(out)
        
        out = self.l2(out)
        out = self.dropout2(out) 
        out = self.relu(out)

        out = self.l3(out)
        # out = self.relu(out)

        # out = self.l4(out)
        # out = self.relu(out)

        # out = self.l5(out)
        out = self.softmax(out)

        return out