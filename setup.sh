#!/bin/bash
echo "Setting up environment..."

# Создаем необходимые директории
mkdir -p models logs data/processed_data data/raw_data

# Устанавливаем права
chmod +x main.py

echo "Setup complete!"