#!/bin/bash
echo "Setting up environment..."

# Создаем необходимые директории
mkdir -p models logs data

# Устанавливаем права
chmod +x main.py

# Создаем необходимые файлы если их нет
touch bot.log

echo "Setup complete!"
