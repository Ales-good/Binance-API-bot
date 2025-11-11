FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib \
    && ldconfig

# Установка Python пакетов в правильном порядке
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install numpy==1.24.3  # Конкретная версия numpy
RUN pip install TA-Lib==0.4.28
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p models logs data

CMD ["python", "main.py"]
