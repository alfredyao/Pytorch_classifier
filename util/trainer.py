import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:

    def __init__(self, model, dataset, batch_size=32, learning_rate=1e-4 ):
        self.model = model
        self.dataset = dataset
        self.batch_size =batch_size 
        self.learning_rate = learning_rate

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model.to(self.device)

        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self,epochs=100):
        self.model.train()

        for epoch in range(epochs):

            for input,target in self.dataloader:

                input = input.to(self.device)
                target = target.to(self.device)

                output = self.model(input)

                loss = self.criterion(output,target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.16f}')

        
        














