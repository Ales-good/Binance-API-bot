FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей включая TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib \
    && ldconfig \
    && apt-get clean

COPY requirements.txt .
RUN pip install --upgrade pip

# Сначала устанавливаем совместимый numpy
RUN pip install "numpy<1.25"

RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p models logs data

CMD ["python", "main.py"]
