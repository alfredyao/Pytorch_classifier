import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):

    def __init__(self, input_dim=16, hidden_dim=20, num_classes=2):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,num_classes)

    def forward(self,X):
        logits = self.fc2(F.relu(self.fc1(X)))

        return logits
    

 