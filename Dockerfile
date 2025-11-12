FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем пакеты по одному для надежности
RUN pip install --upgrade pip

# Базовые пакеты
RUN pip install pandas numpy matplotlib scikit-learn requests python-dotenv

# Торговые пакеты
RUN pip install python-binance websocket-client

# Машинное обучение
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Анализ и визуализация
RUN pip install optuna ta rich mplfinance plotly

# SHAP и LIME (опционально, можно закомментировать если проблемы)
RUN pip install shap lime

COPY . .
RUN mkdir -p models logs data

CMD ["python", "main.py"]
