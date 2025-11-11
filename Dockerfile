FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей для TA-Lib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    build-essential \
    wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/ \
    && ldconfig

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p models logs data

CMD ["python", "main.py"]
