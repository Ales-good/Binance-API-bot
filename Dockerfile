FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements первым (для кэширования)
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копирование остальных файлов
COPY . .

# Создание необходимых директорий
RUN mkdir -p models logs data

# Запуск приложения
CMD ["python", "main.py"]
