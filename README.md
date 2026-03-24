# ScannerPneuIA
Repositório da solução de Scanner da vida útil de um pneu

# 🛞 TireScan AI: Diagnóstico Inteligente de Pneus

![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Data Science](https://img.shields.io/badge/AI%20%26%20Vis%C3%A3o%20Computacional-Ativo-brightgreen)

## 📖 Sobre o Projeto

O **TireScan AI** é uma solução web inovadora que utiliza Inteligência Artificial e Visão Computacional para diagnosticar a vida útil de pneus de carros. Através da câmera do smartphone, o sistema analisa a banda de rodagem, identifica o desgaste (incluindo o indicador TWI) e estima o tempo ou quilometragem restante para a próxima troca.

Este projeto visa não apenas garantir a segurança dos motoristas, mas também atuar como um módulo inteligente que pode ser perfeitamente integrado a ecossistemas maiores, como um marketplace de oficinas mecânicas, conectando a necessidade de manutenção detectada pela IA com os prestadores de serviço.

## ✨ Funcionalidades

* 📸 **Captura Web:** Interface web responsiva que acessa a câmera do dispositivo diretamente pelo navegador, sem necessidade de instalar aplicativos.
* 🧠 **Análise por Visão Computacional:** Processamento da imagem em tempo real utilizando modelos de IA para detectar o nível de desgaste dos sulcos do pneu.
* 📊 **Diagnóstico e Estimativa:** Cálculo algorítmico que cruza o desgaste visual com estimativas de vida útil (tempo/km).
* 💾 **Histórico de Avaliações:** Registro de diagnósticos anteriores salvos em banco de dados para acompanhamento do usuário.

## 🛠️ Tecnologias Utilizadas

A arquitetura do projeto foi pensada para unir performance no backend com um frontend leve:

**Backend & Data Science:**
* **Linguagem:** Python
* **Framework API:** FastAPI (ou Flask)
* **IA / Visão Computacional:** TensorFlow / PyTorch / OpenCV (para processamento de imagem e inferência)
* **Banco de Dados:** SQL (PostgreSQL/MySQL) para estruturação do histórico de escaneamentos e perfis de usuários.

**Frontend:**
* **Web:** React.js / HTML5 + CSS3 + JavaScript puro
* **Integração:** WebRTC API (para acesso nativo à câmera pelo browser)

## 🚀 Como Executar o Projeto (Ambiente de Desenvolvimento)

### Pré-requisitos
* Python 3.10+
* Git
* (Opcional) Ambiente virtual configurado (`venv` ou `conda`)

### Passos para Instalação

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU_USUARIO/tirescan-ai.git](https://github.com/SEU_USUARIO/tirescan-ai.git)
   cd tirescan-ai