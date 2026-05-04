from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Nome do sistema
app = FastAPI(title="ScannerPneuIA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Banco de Dados
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

# 3. Regras de Negócio e Cálculo de Vida Útil (Sprint 4)
# As chaves estão em minúsculo para bater exatamente com o dicionário CLASSES da sua IA
DIAGNOSTICO_REGRAS = {
    "bom": {
        "km_restante": "Aprox. 30.000 km",
        "dicas": ["Mantenha a calibragem quinzenal.", "Faça rodízio a cada 10.000 km."]
    },
    "atencao": {
        "km_restante": "Entre 5.000 km e 10.000 km",
        "dicas": ["Verifique o alinhamento e balanceamento.", "Prepare-se para a troca nos próximos meses."]
    },
    "careca": {
        "km_restante": "0 km (Troca Imediata)",
        "dicas": ["Risco extremo de aquaplanagem e multas.", "Substitua o pneu imediatamente.", "Não pegue estrada."]
    }
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lê e processa a imagem
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    img_tensor = transform(image).unsqueeze(0)
    
    # Faz a predição com o modelo real PyTorch
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confianca, idx = torch.max(probabilities, 0)
        
    resultado = CLASSES[idx.item()]
    
    # Salva no Banco de Dados
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO analises (data, resultado, confianca) VALUES (?, ?, ?)",
                   (datetime.datetime.now().strftime("%d/%m/%Y %H:%M"), resultado, float(confianca)))
    conn.commit()
    conn.close()
    
    # Busca as estimativas e recomendações baseadas no resultado da IA
    # Usamos .get() por segurança, caso a classe não seja encontrada no dicionário
    dados_diagnostico = DIAGNOSTICO_REGRAS.get(resultado, {
        "km_restante": "Indisponível", 
        "dicas": ["Consulte um especialista automotivo."]
    })
    
    # Retorna o JSON estruturado para o Frontend
    return {
        "status": resultado.capitalize(), # Capitalize para ficar "Bom", "Atencao", "Careca" no Frontend
        "confianca": f"{float(confianca)*100:.2f}%",
        "km_estimado": dados_diagnostico["km_restante"],
        "dicas": dados_diagnostico["dicas"]
    }