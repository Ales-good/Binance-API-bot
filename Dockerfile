FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements первыми для кэширования
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копируем остальные файлы
COPY . .

# Создаем необходимые директории
RUN mkdir -p models logs data

CMD ["python", "main.py"]
