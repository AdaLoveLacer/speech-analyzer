#!/bin/bash
cd /home/labubu/projetos-LM

# Cria venv automaticamente se não existir
if [ ! -d "venv" ]; then
    echo "🚀 Criando virtual environment..."
    python -m venv venv
fi

# Ativa venv
source venv/bin/activate

# Instala dependências do requirements.txt (se necessário)
pip install -r requirements.txt --quiet 2>/dev/null || {
    echo "⚠️ Erro ao instalar dependências, tentando instalação manual..."
    pip install flask==3.0.0 openai-whisper python-dotenv openai flask-cors --quiet
}

# Mata processo existente se houver
pkill -f "python app.py" 2>/dev/null || true
sleep 1

export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cria arquivo de persistência se não existir
if [ ! -f ".lmstudio_config.json" ]; then
    echo "Criando arquivo de configuração..."
    touch .lmstudio_config.json
fi
pip install torch --quiet 2>/dev/null || true

echo "Iniciando Whisper Web UI..."
echo "Acessar: http://localhost:5000 ou http://192.168.18.224:5000"
python app.py
