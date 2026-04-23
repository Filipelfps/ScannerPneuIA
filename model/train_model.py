import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# 1. Configurações
DATA_DIR = './dataset' # Certifique-se de que esta pasta existe e tem as imagens!
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. Pré-processamento e Augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carregar imagens
dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
class_names = dataset.classes

# 3. Construção do Modelo (Transfer Learning com MobileNetV2)
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Congela os pesos base
for param in model.parameters():
    param.requires_grad = False

# Troca a última camada para o nosso número de classes
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
model = model.to(DEVICE)

# 4. Compilação
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

# 5. Treinamento
print("Iniciando treinamento da v1 com PyTorch...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataset)
    print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}')

# 6. Exportar Modelo
torch.save(model.state_dict(), 'pneu_model_v1.pth')
print("Modelo exportado com sucesso como 'pneu_model_v1.pth'")