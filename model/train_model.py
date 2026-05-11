import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import random_split, DataLoader
import os

# 1. Configurar transformações das imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Redimensiona para o padrão da MobileNetV2
    transforms.RandomHorizontalFlip(), # Vira algumas imagens para dar mais variedade
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Apontar para a sua pasta exata
data_dir = 'dataset' # Como o script está dentro da pasta 'model', ele vai achar a 'dataset' do lado dele

# Carregar todas as imagens das 3 pastas
full_dataset = datasets.ImageFolder(data_dir, transform=transform)

# 3. Dividir automaticamente (80% treino, 20% validação)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Criar os Loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 4. Carregar MobileNetV2 e ajustar para 3 classes
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False # Congela a base

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 3) # Nossas 3 classes: atencao, bom, careca
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# AQUI ENTRA O SEU LOOP DE TREINAMENTO (for epoch in range(num_epochs): ...)

# 5. Salvar o novo modelo sobrescrevendo o antigo
torch.save(model.state_dict(), 'pneu_model_v1.pth')
print("Treinamento concluído. pneu_model_v1.pth atualizado!")