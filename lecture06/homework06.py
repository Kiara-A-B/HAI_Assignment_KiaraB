import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                 
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)      
        )

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = myModel()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10

for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss_value = loss_fn(outputs, labels)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item()

    print(f"Epoch [{epoch:02d} Train Loss: {total_loss/len(train_loader):.4f}")

img = Image.open("/Users/kiarabauerle/Documents/Hanyang/AI study group/HAI_Assignment_KiaraB/lecture06/numberImage.png").convert("L")
to_tensor = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

model.eval()
x = to_tensor(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
    probs = nn.functional.softmax(logits, dim=1)

print(probs)