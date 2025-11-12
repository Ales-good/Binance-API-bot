FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip

# Сначала устанавливаем все пакеты кроме PyTorch
RUN pip install pandas numpy matplotlib scikit-learn requests python-dotenv \
                python-binance websocket-client optuna ta rich mplfinance plotly

# Затем устанавливаем PyTorch с отдельным index
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN mkdir -p models logs data

CMD ["python", "main.py"]
