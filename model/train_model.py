import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import random_split, DataLoader
import os

# 1. Transformações
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Carregar Dataset
data_dir = 'dataset' 
full_dataset = datasets.ImageFolder(data_dir, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 3. Preparar MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False 

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 3) 
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 4. O LOOP DE TREINAMENTO REAL (10 Épocas)
print("Iniciando o treinamento de verdade. Isso vai levar alguns minutos...")
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    print(f"Época {epoch+1}/{num_epochs} concluída - Erro (Loss): {epoch_loss:.4f}")

# 5. Salvar o modelo treinado
torch.save(model.state_dict(), 'pneu_model_v1.pth')
print("Treinamento concluído com sucesso! pneu_model_v1.pth atualizado com inteligência real.")