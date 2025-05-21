import torch
import torch.nn as nn
import torch.optim as optim

class network (nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(17, 10),   
            nn.ReLU(),
            nn.Linear(10, 5),   
            nn.ReLU(),
            nn.Linear(5, 1)      
        )
    def forward(self, x):
        return self.model(x)

model = network()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs = torch.randn(100, 17)

targets = torch.randn(100, 1)

for epoch in range(100):

    outputs = model(inputs)

    loss = criterion(outputs, targets)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], loss: {loss.item():.4f}")