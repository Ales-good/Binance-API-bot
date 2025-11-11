FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей включая TA-Lib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Установка TA-Lib зависимостей и компиляция
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

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
