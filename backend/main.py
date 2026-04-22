from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# CORREÇÃO AQUI: Nome do sistema
app = FastAPI(title="ScannerPneuIA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Banco de Dados (Nome corrigido)
DB_FILE = "scanner_pneu.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS analises 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT, resultado TEXT, confianca REAL)''')
    conn.commit()
    conn.close()

init_db()

# 2. Carrega a IA
CLASSES = {0: 'atencao', 1: 'bom', 2: 'careca'}
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))

try:
    # Busca o modelo na pasta model que você treinou
    model.load_state_dict(torch.load('model/pneu_model_v1.pth'))
    model.eval()
    print("✅ IA do ScannerPneuIA carregada com sucesso!")
except Exception as e:
    print(f"⚠️ Erro ao carregar modelo: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confianca, idx = torch.max(probabilities, 0)
        
    resultado = CLASSES[idx.item()]
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO analises (data, resultado, confianca) VALUES (?, ?, ?)",
                   (datetime.datetime.now().strftime("%d/%m/%Y %H:%M"), resultado, float(confianca)))
    conn.commit()
    conn.close()
    
    return {"status": resultado, "confianca": f"{confianca*100:.2f}%"}