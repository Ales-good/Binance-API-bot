#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 1. Базовые импорты
import os
import sys
import io
import math
import time
import signal
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 2. Импорты Rich для логирования
from rich.console import Console
from rich.logging import RichHandler

# Добавьте эти импорты сюда:
import shap
import lime
import lime.lime_tabular

# 11. Для Backtesting
from typing import Dict, List
from datetime import datetime

# 12. Асинхронные методы обучения модели
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 3. Определяем SafeConsole перед использованием
class SafeConsole(Console):
    """Переопределенный Console с обработкой ошибок вывода"""
    def print(self, *args, **kwargs):
        try:
            super().print(*args, **kwargs)
        except (ValueError, AttributeError):
            sys.stdout.write(str(args) + "\n")

# 4. Настройка логирования
def configure_logging():
    """Безопасная настройка системы логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler('bot.log', encoding='utf-8', mode='w'),
            RichHandler(
                console=SafeConsole(force_terminal=False),
                show_path=False,
                rich_tracebacks=True
            )
        ],
        force=True
    )
    logging.captureWarnings(True)

# 5. Инициализация логгера и консоли
configure_logging()
logger = logging.getLogger(__name__)
console = SafeConsole()  # Теперь SafeConsole определен

# 6. Остальные импорты (после инициализации логгера)
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from binance import ThreadedWebsocketManager
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from io import BytesIO
from matplotlib.animation import FuncAnimation
import optuna
from sklearn.model_selection import TimeSeriesSplit

# 7. Фиксы для Windows
if sys.platform == "win32":
    # Принудительная установка UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ["PYTHONIOENCODING"] = "utf-8"

# 8. Отключаем matplotlib GUI
import matplotlib
matplotlib.use('Agg')  # Неинтерактивный бэкенд
import matplotlib.pyplot as plt

# Railway-specific fixes
import os
if 'RAILWAY_ENVIRONMENT' in os.environ:
    # Отключаем GUI полностью
    import matplotlib
    matplotlib.use('Agg')
    
    # Настройка путей для Railway
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
# TA-Lib fallback with ta alternative
try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("TA-Lib successfully imported")
except ImportError:
    try:
        # Используем ta как альтернативу
        import ta
        TALIB_AVAILABLE = False
        TA_AVAILABLE = True
        logger.info("TA-Lib not available, using 'ta' library instead")
    except ImportError:
        TALIB_AVAILABLE = False
        TA_AVAILABLE = False
        logger.warning("Neither TA-Lib nor ta available, using pure Python implementations")
    
    # Pure Python implementations
    def calculate_rsi_python(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(period).mean()
        avg_losses = pd.Series(losses).rolling(period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd_python(prices, fast=12, slow=26, signal=9):
        ema_fast = pd.Series(prices).ewm(span=fast).mean()
        ema_slow = pd.Series(prices).ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line, macd - signal_line
    
    def calculate_atr_python(high, low, close, period=14):
        tr = np.maximum(high - low, 
                       np.maximum(np.abs(high - close), 
                                 np.abs(low - close)))
        atr = pd.Series(tr).rolling(period).mean()
        return atr

# ... остальной код классов и функций


class HyperparameterOptimizer:
    def __init__(self, bot):
        self.bot = bot
        self.scalers = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def objective(self, trial):
        print(f"Shapes before training - x: {x.shape}, y: {y.shape}")  # Должно быть (N, features) и (N,)
        assert y.ndim == 1, f"Y must be 1D, got {y.shape}"
        params = {
            'model_dim': trial.suggest_int('model_dim', 64, 256),
            'num_heads': trial.suggest_int('num_heads', 2, 8),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5)
        }
    
        # 1. Получение и проверка данных
        x, y, _ = self.bot._prepare_transformer_data()
        if x is None or len(x) < 100:
            return float('inf')
    
        # 2. Преобразование в numpy с проверкой размеров
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
    
        assert x.ndim == 3, f"Ожидается 3D массив [samples, seq_len, features], получено {x.shape}"
        assert y.ndim == 1, f"Ожидается 1D массив [samples], получено {y.shape}"
    
        # 3. Разделение данных
        tscv = TimeSeriesSplit(n_splits=3)
        val_losses = []
    
        for train_idx, val_idx in tscv.split(x):
            # 4. Подготовка тензоров
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        
            x_train_tensor = torch.from_numpy(x_train).float().to(self.device)
            y_train_tensor = torch.from_numpy(y_train).float().to(self.device)
            x_val_tensor = torch.from_numpy(x_val).float().to(self.device)
            y_val_tensor = torch.from_numpy(y_val).float().to(self.device)
        
            # 5. Инициализация модели с текущими параметрами
            model = TransformerModel(
                input_dim=x.shape[-1],
                model_dim=params['model_dim'],
                num_heads=params['num_heads'],
                num_layers=params['num_layers'],
                output_dim=1
            ).to(self.device)
        
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.MSELoss()
        
            # 6. Обучение с проверкой размеров
            model.train()
            for epoch in range(10):
                optimizer.zero_grad()
            
                outputs = model(x_train_tensor)
                assert outputs.shape == y_train_tensor.shape, \
                    f"Несоответствие размеров: outputs {outputs.shape}, y {y_train_tensor.shape}"
            
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
        
            # 7. Валидация
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_losses.append(val_loss.item())
    
            return np.mean(val_losses)

        # Гарантируем 2D форму
        y = y.reshape(-1, 1)
    
        tscv = TimeSeriesSplit(n_splits=3)
        val_losses = []
    
        for train_idx, val_idx in tscv.split(x):
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        
            # Преобразование с явным контролем формы
            x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(self.device)
        
            model = TransformerModel(
                input_dim=x.shape[-1],
                model_dim=params['model_dim'],
                num_heads=params['num_heads'],
                num_layers=params['num_layers'],
                output_dim=1
            ).to(self.device)
        
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.MSELoss()
        
            # Обучение
            model.train()
            for epoch in range(10):
                optimizer.zero_grad()
    
                # Явное преобразование с контролем формы
                x_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
                y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(self.device)  # Критично: reshape вместо unsqueeze
    
                outputs = model(x_tensor)
    
                # Принудительное выравнивание форм
                outputs = outputs.view(-1, 1)
                y_tensor = y_tensor.view(-1, 1)
    
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
        
            # Валидация
            model.eval()
            with torch.no_grad():
                x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(self.device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(self.device)
    
                val_outputs = model(x_val_tensor).view(-1, 1)
                y_val_tensor = y_val_tensor.view(-1, 1)
    
                val_loss = criterion(val_outputs, y_val_tensor)
                val_losses.append(val_loss.item())
    
        return np.mean(val_losses)
        
    def optimize(self, n_trials=50):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        # Сохраняем лучшие параметры
        best_params = study.best_params
        logger.info(f"Лучшие параметры: {best_params}")
        
        # Обновляем модель бота
        self.bot.transformer_model = TransformerModel(
            input_dim=10,
            model_dim=best_params['model_dim'],
            num_heads=best_params['num_heads'],
            num_layers=best_params['num_layers'],
            output_dim=1
        )
        
        return best_params

class SafeConsole(Console):
    """Переопределенный Console с обработкой ошибок вывода"""
    def print(self, *args, **kwargs):
        try:
            super().print(*args, **kwargs)
        except (ValueError, AttributeError):
            sys.stdout.write(str(args) + "\n")

    def configure_logging():
        """Безопасная настройка системы логирования"""
        # Создаем кастомный обработчик для Rich
        class SafeRichHandler(RichHandler):
            def emit(self, record):
                try:
                    super().emit(record)
                except:
                    # Fallback на базовое логирование
                    sys.stderr.write(f"{record.levelname}: {record.msg}\n")

        # Конфигурация
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler('bot.log', encoding='utf-8', mode='w'),
                SafeRichHandler(
                    console=SafeConsole(force_terminal=False),
                    show_path=False,
                    rich_tracebacks=True
                )
            ],
            force=True
        )
        logging.captureWarnings(True)

        # Инициализация
        configure_logging()
        logger = logging.getLogger(__name__)
        console = SafeConsole()
import shap
import lime
import lime.lime_tabular

def analyze_feature_importance(model, x_train, feature_names):
    """
    Анализ важности фичей через SHAP и LIME.
    
    Args:
        model: Обученная PyTorch-модель.
        x_train: Тренировочные данные (np.array).
        feature_names: Список названий фичей.
    """
    try:
        # SHAP
        explainer_shap = shap.DeepExplainer(model, x_train[:100])  # Используем подвыборку
        shap_values = explainer_shap.shap_values(x_train[:10])
        
        # Визуализация SHAP
        shap.summary_plot(shap_values, x_train[:10], feature_names=feature_names)
        plt.savefig("shap_summary.png")
        plt.close()
        
        # LIME
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            x_train.mean(axis=0),
            feature_names=feature_names,
            mode="regression"
        )
        exp = explainer_lime.explain_instance(x_train[0], model.predict)
        exp.save_to_file("lime_explanation.html")
        
        logger.info("Анализ важности фичей завершен. Результаты сохранены.")
        
    except Exception as e:
        logger.error(f"Ошибка анализа фичей: {str(e)}", exc_info=True)
class Backtester:
    """Класс для тестирования стратегии на исторических данных"""
    
    def __init__(self, bot):
        self.bot = bot  # Ссылка на основной бот
        self.results = []
        self.equity_curve = []
        
    def run_backtest(self, start_date: str, end_date: str):
        """Запуск тестирования на исторических данных"""
        try:
            # Загрузка данных
            klines = self.bot.client.futures_klines(
                symbol=self.bot.symbol,
                interval=self.bot.interval,
                startTime=int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000),
                endTime=int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000),
                limit=1000
            )
            
            # Имитация работы бота на исторических данных
            for kline in klines:
                candle = {
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                }
                
                # Обновляем данные бота
                self.bot.data = pd.concat([
                    self.bot.data, 
                    pd.DataFrame([candle])
                ], ignore_index=True).tail(self.bot.data_window_size)
                
                # Рассчитываем индикаторы
                indicators = self.bot._calculate_indicators()
                
                # Логика входа/выхода (без реальных сделок)
                if not self.bot.current_position:
                    if self.bot._check_entry_signal(indicators):
                        side = 'BUY' if indicators['ema5'] > indicators['ema10'] else 'SELL'
                        self._simulate_trade(candle['close'], side)
                else:
                    if self.bot._check_exit_signal(indicators):
                        self._simulate_trade(candle['close'], 'CLOSE')
                        
                # Записываем equity (для графика)
                self.equity_curve.append({
                    'timestamp': kline[0],
                    'equity': self.bot.current_balance
                })
                
                # Анализ результатов
                self._analyze_results()
            
        except Exception as e:
            self.bot.logger.error(f"Ошибка backtest: {e}", exc_info=True)
    
    def _simulate_trade(self, price: float, action: str):
        """Имитация сделки (без реального API)"""
        if action == 'BUY':
            self.bot.current_position = {
                'side': 'BUY',
                'entry_price': price,
                'size': self.bot._calculate_position_size(price),
                'opened_at': datetime.now()
            }
            self.bot.logger.info(f"[BACKTEST] Открыта LONG позиция по {price}")
            
        elif action == 'SELL':
            self.bot.current_position = {
                'side': 'SELL',
                'entry_price': price,
                'size': self.bot._calculate_position_size(price),
                'opened_at': datetime.now()
            }
            self.bot.logger.info(f"[BACKTEST] Открыта SHORT позиция по {price}")
            
        elif action == 'CLOSE':
            pnl = self._calculate_pnl(price)
            self.bot.current_balance += pnl
            self.results.append({
                'side': self.bot.current_position['side'],
                'pnl': pnl,
                'duration': (datetime.now() - self.bot.current_position['opened_at']).total_seconds() / 60
            })
            self.bot.current_position = None
            self.bot.logger.info(f"[BACKTEST] Закрыта позиция. PnL: {pnl:.2f} USDT")
    
    def _calculate_pnl(self, exit_price: float) -> float:
        """Расчет прибыли/убытка"""
        entry_price = self.bot.current_position['entry_price']
        size = self.bot.current_position['size']
        
        if self.bot.current_position['side'] == 'BUY':
            return (exit_price - entry_price) * size
        else:
            return (entry_price - exit_price) * size
    
    def _analyze_results(self):
        """Анализ результатов тестирования"""
        if not self.results:
            self.bot.logger.warning("[BACKTEST] Нет сделок для анализа")
            return
            
        df = pd.DataFrame(self.results)
        win_rate = (df['pnl'] > 0).mean() * 100
        avg_pnl = df['pnl'].mean()
        total_pnl = df['pnl'].sum()
        
        self.bot.logger.info(
            f"[BACKTEST] Результаты:\n"
            f"┌ Сделок: {len(df)}\n"
            f"├ Win Rate: {win_rate:.1f}%\n"
            f"├ Средний PnL: {avg_pnl:.2f} USDT\n"
            f"└ Общий PnL: {total_pnl:.2f} USDT"
        )
        
        # Визуализация equity curve
        self._plot_equity_curve()
    
    def _plot_equity_curve(self):
        """График изменения баланса"""
        import matplotlib.pyplot as plt
        df = pd.DataFrame(self.equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['equity'], label='Equity')
        plt.title("Backtest Equity Curve")
        plt.xlabel("Дата")
        plt.ylabel("Баланс (USDT)")
        plt.grid(True)
        plt.legend()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        
        self.bot._send_telegram_alert("[BACKTEST] Результаты тестирования", buf)




class PositionalEncoding(nn.Module):
    """Реализация позиционного кодирования для трансформеров.
    Args:
    d_model: Размерность эмбеддингов
    max_len: Максимальная длина последовательности (по умолчанию 5000)"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Предварительное вычисление div_term
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(max_len, d_model)
        
        # Заполнение четных и нечетных индексов
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Регистрация как буфера (не обучаемого параметра)
        self.register_buffer('pe', pe.unsqueeze(0))  # Добавляем размерность батча

        # Добавляем новые параметры
        self.sequence_length = 60  # Длина последовательности для Transformer
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'atr', 'obv', 'vwap'
        ]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Добавляет позиционное кодирование к входным данным.
        
        Args:
            x: Входной тензор формы [batch_size, seq_len, d_model]
            
        Returns:
            Тензор с добавленным позиционным кодированием
        """
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Размерность входных данных ({x.size(-1)}) "
                f"не соответствует d_model ({self.d_model})"
            )
            
        if x.size(1) > self.max_len:
            raise ValueError(
                f"Длина последовательности ({x.size(1)}) "
                f"превышает max_len ({self.max_len})"
            )
            
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim=10, model_dim=128, num_heads=8, num_layers=4, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Исправленная последовательность слоёв
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, output_dim)
        )  # Закрывающая скобка была пропущена

    def forward(self, src):
        # Проверка и корректировка входной размерности
        if src.dim() == 2:
            src = src.unsqueeze(0)
        elif src.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {src.dim()}D")
            
        # Проход через слои модели
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer(src)
        output = self.fc(output[:, -1, :])  # Берем последний элемент последовательности
        
        return output.squeeze(-1)  # Возвращаем [batch_size] для совместимости с MSE
    
class EarlyStopping:
    """Ранняя остановка для обучения модели"""
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

class AggressiveFuturesBot:
    def _init_new_model(self):
        """Инициализирует новую модель с фиксированными параметрами"""
        try:
            logger.info("Создаю новую модель и scaler")
        
            # Фиксированные параметры модели (должны совпадать при сохранении/загрузке)
            model_params = {
                'input_dim': 10,  # Количество фичей
                'model_dim': 128,
                'num_heads': 4,
                'num_layers': 2,   # Уменьшите если были проблемы с 4 слоями
                'output_dim': 1
            }
        
            self.transformer_model = TransformerModel(**model_params)
            self.scaler = MinMaxScaler()
        
            logger.info(f"Новая модель создана с параметрами: {model_params}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка создания модели: {str(e)}")
            return False
            
    def __init__(self):
        """Инициализация агрессивного торгового бота"""
        self.executor = ThreadPoolExecutor(max_workers=2)  # Пул потоков
        self.loop = asyncio.get_event_loop()
        try:
            load_dotenv()
            self._check_env_vars()
            
            # Проверяем API ключи перед созданием клиента
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET_KEY')
            
            if not api_key or not secret_key:
                raise ValueError("API ключи не найдены в переменных окружения")
            
            logger.info(f"API Key: {api_key[:10]}...")
            
            # Пробуем подключиться к Binance
            try:
                self.client = Client(api_key, secret_key, testnet=False)
                # Тестовый запрос для проверки подключения
                self.client.futures_exchange_info()
                logger.info("✅ Подключение к Binance Futures успешно")
                
                # Проверяем баланс фьючерсного аккаунта
                time.sleep(1)
                account_info = self.client.futures_account()
                balance = float(account_info['totalWalletBalance'])
                
                if balance <= 0:
                    logger.warning(f"⚠️ На фьючерсном аккаунте нет средств: {balance} USDT")
                    logger.info("💡 Пополните фьючерсный аккаунт через Binance App/Website")
                    # Продолжаем работу, но предупреждаем что торговля невозможна
                else:
                    logger.info(f"✅ Баланс фьючерсного аккаунта: {balance:.2f} USDT")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка подключения к Futures: {e}")
                # Пробуем testnet для диагностики
                try:
                    logger.info("🔄 Пробуем подключиться к Testnet...")
                    self.client = Client(api_key, secret_key, testnet=True)
                    self.client.futures_exchange_info()
                    logger.info("✅ Подключение к Binance Testnet успешно")
                    logger.warning("⚠️ Используется TESTNET для отладки")
                except Exception as testnet_error:
                    logger.error(f"❌ Ошибка подключения к Testnet: {testnet_error}")
                    raise ValueError("Не удалось подключиться ни к mainnet, ни к testnet")
            
            self.ws_manager = ThreadedWebsocketManager(
                api_key=api_key,
                api_secret=secret_key
            )
            
            # Агрессивные параметры торговли
            self.symbol = "BTCUSDT"
            self.leverage = 30  # среднее плечо
            self.interval = Client.KLINE_INTERVAL_5MINUTE  # Более короткий таймфрейм
            self.risk_percent = 0.3  # Средний риск на сделку
            self.take_profit = 0.01  # 1.0% тейк-профит
            self.stop_loss = 0.01  # 1.0% стоп-лосс
            self.max_retries = 3
            self.retry_delay = 2  # Уменьшенная задержка между попытками
            self.data_window_size = 1500  # Оптимальный размер окна данных
            self.min_training_samples = 3500  # Минимум свечей для обучения
            
            # Состояние бота
            self.current_position = None
            self.data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            self.starting_balance = 0.0
            self.current_balance = 0.0
            self.equity = 0.0
            self.trade_count = 0
            self.profit_loss = 0.0
            self.win_rate = 0.0
            self.win_count = 0
            self.ws_conn = None
            self.last_candle_time = None
            self.transformer_model = None
            self.scaler = None
            
            # Инициализация модели - только ОДИН раз
            try:
                if self.load_model():
                    logger.info("✅ Модель успешно загружена")
                else:
                    logger.warning("Файл модели не найден, создаю новую модель")
                    self._init_new_model()
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель: {str(e)}")
                self._init_new_model()
            
            self.min_qty = 0.001  # Минимальный размер ордера для BTCUSDT
            
            # Настройка обработчиков сигналов
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            
            logger.info("Агрессивный-осторожный бот инициализирован")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации: {str(e)}", exc_info=True)
            self._send_telegram_alert(f"❌ Ошибка инициализации агрессивного бота: {str(e)}")
            raise

    def load_historical_data(self, days=30):
        """Загрузка исторических данных с пагинацией и обработкой таймаутов"""
        try:
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - days * 24 * 60 * 60 * 1000
    
            all_klines = []
            current_end = end_time
    
            # Увеличиваем таймаут для клиента
            self.client.timeout = 30  # Увеличиваем до 30 секунд
        
            # Пагинация: получаем данные порциями по 1000 свечей
            while current_end > start_time:
                try:
                    klines = self._retry_api_call(
                        self.client.futures_klines,
                        symbol=self.symbol,
                        interval=self.interval,
                        limit=1000,
                        endTime=current_end,
                        timeout=30  # Увеличиваем таймаут для запроса
                    )
            
                    if not klines:
                        break
                
                    all_klines.extend(klines)
                    current_end = klines[0][0] - 1  # Время начала предыдущей свечи
            
                    # Задержка для избежания лимитов API
                    time.sleep(0.5)  # Увеличиваем задержку между запросами
            
                except Exception as e:
                    logger.error(f"Ошибка при загрузке данных: {str(e)}")
                    time.sleep(5)  # Большая задержка при ошибке
                    continue
    
            if not all_klines:
                raise ValueError("Не удалось получить данные")
        
            # Формируем DataFrame
            self.data = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
    
            # Конвертация типов
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            self.data[numeric_cols] = self.data[numeric_cols].astype(float)
    
            logger.info(f"Загружено {len(self.data)} свечей (за {days} дней)")
            return True
    
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {str(e)}", exc_info=True)
            return False
    
    async def train_model_async(self):
        """Асинхронное обучение модели"""
        try:
            result = await self.loop.run_in_executor(
                self.executor,
                self._train_transformer_model
            )
            if result:
                await self.loop.run_in_executor(
                    self.executor,
                    self.save_model
                )
            return result
        except Exception as e:
            self.logger.error(f"Ошибка асинхронного обучения: {e}", exc_info=True)
            return False

    async def predict_async(self, price: float):
        """Асинхронный прогноз"""
        try:
            prediction = await self.loop.run_in_executor(
                self.executor,
                self._predict_with_transformer,
                price
            )
            return prediction
        except Exception as e:
            self.logger.error(f"Ошибка асинхронного прогноза: {e}", exc_info=True)
            return None
        
    def run_backtest(self, start_date: str, end_date: str):
        """Запуск тестирования стратегии"""
        backtester = Backtester(self)
        backtester.run_backtest(start_date, end_date)
    
    def _check_internet(self):
        """Проверка интернет-соединения"""
        try:
            response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
            return response.status_code == 200
        except:
            return False
        
    def verify_api_keys(self):
        """Проверка валидности API-ключей"""
        try:
            self.client.get_account()
            return True
        except Exception as e:
            logger.error(f"Ошибка ключей API: {str(e)}")
            return False
    
    def check_api_status(self):
        """Проверка статуса Binance API"""
        try:
            status = self.client.get_system_status()
            return status['status'] == 0
        except:
            return False
    
    def is_ip_banned(self):
        """Проверка блокировки IP"""
        try:
            self.client.get_exchange_info()
            return False
        except BinanceAPIException as e:
            return e.status_code == 403
    
    def check_connection(self):
        """Комплексная проверка соединения"""
        checks = {
            'Интернет': self._check_internet(),
            'API Keys': self.verify_api_keys(),
            'Статус API': self.check_api_status(),
            'IP блокировка': not self.is_ip_banned()
        }

        if not all(checks.values()):
            error_msg = "Проблемы соединения:\n" + "\n".join(
                f"{k}: {'✔' if v else '✖'}" for k, v in checks.items()
            )
            logger.error(error_msg)
            self._send_telegram_alert(error_msg)
            return False
        return True

    def _check_websocket(self):
        """Проверка состояния WebSocket"""
        if not self.ws_manager or not self.ws_manager.is_alive():
            logger.error("WebSocket не работает")
            return False
        return True
    
    def _validate_data(self):
        """Проверка качества данных"""
        if len(self.data) < 50:
            logger.warning("Недостаточно данных для анализа")
            return False
        if self.data.isnull().values.any():
            logger.warning("Обнаружены пропущенные значения в данных")
            return False
        return True
    
    def _handle_shutdown(self, signum, frame):
        """Обработчик сигналов завершения работы"""
        logger.info(f"Получен сигнал {signum}, завершаю работу...")
        self.stop_websocket()
        sys.exit(0)

    def start_websocket(self):
        """Запуск WebSocket соединения с улучшенной обработкой ошибок"""
        try:
            # Останавливаем предыдущее соединение, если есть
            self.stop_websocket()

            # Инициализируем менеджер с правильными параметрами
            self.ws_manager = ThreadedWebsocketManager(
                api_key=os.getenv('BINANCE_API_KEY'),
                api_secret=os.getenv('BINANCE_SECRET_KEY')
            )
        
            # Настройка параметров соединения (для версии python-binance 1.0.0+)
            if hasattr(self.ws_manager, 'set_socket_timeout'):
                self.ws_manager.set_socket_timeout(30)  # Таймаут в секундах
        
            # Запускаем менеджер
            self.ws_manager.start()
        
            # Запускаем поток для получения данных
            self.ws_conn = self.ws_manager.start_kline_socket(
                symbol=self.symbol,
                interval=self.interval,
                callback=self._handle_websocket_message
            )
        
            logger.info(f"WebSocket успешно запущен для {self.symbol} {self.interval}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка запуска WebSocket: {str(e)}", exc_info=True)
            self._send_telegram_alert(f"❌ WebSocket ошибка: {str(e)[:200]}")
            return False

    def _check_binance_version(self):
        """Проверяет версию python-binance"""
        try:
            import binance
            version = binance.__version__
            logger.info(f"Используется python-binance версии {version}")
            return version
        except Exception as e:
            logger.error(f"Ошибка проверки версии: {str(e)}")
            return "unknown"

    def _handle_websocket_error(self, error):
        """Обработчик ошибок WebSocket"""
        try:
            error_msg = str(error)
            logger.error(f"WebSocket error: {error_msg}")
        
            # Отправляем уведомление только для серьезных ошибок
            if "reconnect" not in error_msg.lower():
                self._send_telegram_alert(f"⚠️ WebSocket ошибка: {error_msg[:200]}...")
            
            # Пытаемся переподключиться
            time.sleep(5)
            self.start_websocket()
        
        except Exception as e:
            logger.error(f"Ошибка в обработчике WebSocket: {str(e)}")

    def is_websocket_connected(self):
        """Проверяет активность WebSocket соединения"""
        try:
            return (hasattr(self, 'ws_manager') and 
                   self.ws_manager and 
                   hasattr(self.ws_manager, 'is_alive') and 
                   self.ws_manager.is_alive())
        except Exception:
            return False

    def stop_websocket(self):
        """Корректная остановка WebSocket соединения"""
        try:
            if hasattr(self, 'ws_manager') and self.ws_manager:
                self.ws_manager.stop()
                logger.info("WebSocket остановлен")
            if hasattr(self, 'ws_conn'):
                del self.ws_conn
        except Exception as e:
            logger.error(f"Ошибка остановки WebSocket: {str(e)}")

    def _check_env_vars(self):
        """Проверка обязательных переменных окружения"""
        required_vars = {
            'BINANCE_API_KEY': 'API ключ Binance',
            'BINANCE_SECRET_KEY': 'Секретный ключ Binance',
            'TELEGRAM_BOT_TOKEN': 'Токен Telegram бота',
            'TELEGRAM_CHAT_ID': 'ID Telegram чата'
        }
        missing_vars = [name for var, name in required_vars.items() if not os.getenv(var)]
        if missing_vars:
            error_msg = f"Отсутствуют переменные окружения: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _send_telegram_alert(self, message: str, chart=None):
        """Отправка сообщений и графиков в Telegram"""
        try:
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
            if not token or not chat_id:
                raise ValueError("Не заданы Telegram credentials")

            if chart:
                # Вариант 1: Отправка как файла (надежнее)
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                files = {'photo': ('chart.png', chart.getvalue())}
                data = {'chat_id': chat_id, 'caption': message[:1024], 'parse_mode': 'HTML'}
                response = requests.post(url, files=files, data=data, timeout=15)
            else:
                # Только текст
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                params = {
                    'chat_id': chat_id,
                    'text': message[:4096],
                    'parse_mode': 'HTML'
                }
                response = requests.post(url, params=params, timeout=10)
        
            response.raise_for_status()
            return True
        
        except Exception as e:
            logger.error(f"Telegram send error: {str(e)}")
            return False

    def _generate_chart(self):
        """Генерация статичного графика (без анимации)"""
        try:
            plt.figure(figsize=(12, 6))
            data = self.data.tail(100)
        
            plt.plot(data['close'], label='Price', color='blue')
            plt.title(f"{self.symbol} Price Chart")
            plt.legend()
            plt.grid(True)
        
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close()
        
            return buf
        
        except Exception as e:
            logger.error(f"Chart generation error: {str(e)}")
            return None

    def _generate_live_chart(self):
        """Генерация анимированного графика с исправлением ошибок"""
        try:
            # Создаем фигуру в неинтерактивном режиме
            plt.ioff()
            fig, ax = plt.subplots(figsize=(12, 6))
        
            # Берем последние 100 свечей
            data = self.data.tail(100)
        
            # Функция анимации
            def animate(i):
                ax.clear()
                ax.plot(data['close'].iloc[:i+1], label='Цена', color='blue')
                ax.set_title(f"Live график {self.symbol}")
                ax.legend()
                ax.grid(True)
        
            # Создаем анимацию
            anim = FuncAnimation(
                fig,
                animate,
                frames=len(data),
                interval=200,
                repeat=False
            )
        
            # Сохраняем в буфер BytesIO
            buf = BytesIO()
            writer = 'pillow'  # Используем pillow для GIF
            anim.save(buf, writer=writer, fps=5)
            buf.seek(0)  # Перемещаем указатель в начало буфера
        
            # Закрываем фигуру
            plt.close(fig)
        
            return buf
        
        except Exception as e:
            logger.error(f"Ошибка генерации live-графика: {str(e)}", exc_info=True)
            if 'fig' in locals():
                plt.close(fig)
            return None

    def _retry_api_call(self, func, *args, **kwargs):
        """Повторная попытка вызова API с коррекцией времени"""
        for attempt in range(self.max_retries):
            try:
                # Добавляем коррекцию временной метки только для методов, которые ее требуют
                if 'timestamp' in kwargs or func.__name__ in ['futures_account', 'futures_create_order', 'futures_change_leverage']:
                    if 'timestamp' not in kwargs:
                        kwargs['timestamp'] = int(time.time() * 1000) - 1000  # Корректируем на 1 секунду назад
                return func(*args, **kwargs)
            except BinanceAPIException as e:
                if e.code == -1021:  # Ошибка временной метки
                    time_diff = self._get_server_time_diff()
                    kwargs['timestamp'] = int(time.time() * 1000) + time_diff
                    continue
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Попытка {attempt + 1} из {self.max_retries} не удалась: {str(e)}")
                time.sleep(self.retry_delay)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Попытка {attempt + 1} из {self.max_retries} не удалась: {str(e)}")
                time.sleep(self.retry_delay)

    def _adjust_leverage_based_on_volatility(self):
        """Динамическое изменение плеча на основе волатильности"""
        atr = self._calculate_atr()
        price = self.data['close'].iloc[-1]
        atr_percent = (atr / price) * 100
    
        if atr_percent < 1.0:
            new_leverage = min(75, self.leverage + 5)
        elif atr_percent > 3.0:
            new_leverage = max(10, self.leverage - 5)
    
        if new_leverage != self.leverage:
            self.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=new_leverage
            )
            self.leverage = new_leverage
            logger.info(f"Плечо изменено на {new_leverage}x")

    def _log_margin_status(self):
        account = self.client.futures_account()
        balance = float(account['totalWalletBalance'])
        available = float(account['availableBalance'])
        used = balance - available
    
        logger.info(
            f"Маржа: Использовано {used:.2f}/{balance:.2f} USDT "
            f"({used/balance*100:.1f}%)"
        )

    def _get_server_time_diff(self):
        """Получаем разницу между локальным временем и временем сервера Binance"""
        try:
            server_time = self.client.get_server_time()['serverTime']
            local_time = int(time.time() * 1000)
            return server_time - local_time
        except Exception as e:
            logger.warning(f"Не удалось получить разницу времени: {str(e)}")
            return -1000  # Возвращаем фиксированную корректировку
    def _update_account_info(self):
        """Обновление информации о счете с проверкой на None"""
        try:
            account = self._retry_api_call(self.client.futures_account)
            if not account or 'assets' not in account:
                logger.error("Не удалось получить данные аккаунта или структура неверна")
                return False
            
            usdt_balance = next((item for item in account['assets'] if item['asset'] == 'USDT'), None)
        
            if not usdt_balance:
                logger.error("Не найден баланс USDT в данных аккаунта")
                return False
            
            self.current_balance = float(usdt_balance['availableBalance'])
            self.equity = float(account['totalWalletBalance'])
        
            if not self.starting_balance:
                self.starting_balance = self.current_balance
            
                logger.info(f"Баланс: {self.current_balance:.2f} USDT | Эквити: {self.equity:.2f} USDT")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка обновления баланса: {str(e)}", exc_info=True)
            self._send_telegram_alert(f"⚠️ Ошибка обновления баланса: {str(e)}")
            return False
    def _check_api_connection(self):
        """Проверка работоспособности API"""
        try:
            if not self.client:
                logger.error("Клиент API не инициализирован")
                return False
            # Простой тестовый запрос
            server_time = self.client.get_server_time()
            if not server_time:
                logger.error("Не удалось получить время сервера")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Ошибка соединения с API: {str(e)}")
            return False
    def _setup_leverage(self):
        """Настройка кредитного плеча с обработкой ошибок времени"""
        try:
            # Получаем текущее время сервера для точной синхронизации
            server_time = self.client.get_server_time()['serverTime']
            time_diff = int(time.time() * 1000) - server_time

            return self._retry_api_call(
                self.client.futures_change_leverage,
                symbol=self.symbol,
                leverage=self.leverage,
                timestamp=server_time - time_diff  # Используем скорректированное время
        )
        except Exception as e:
            logger.error(f"Ошибка установки плеча: {str(e)}", exc_info=True)
            return False
            
    def _get_symbol_info(self):
        """Получение информации о торговой паре"""
        try:
            # Убрали передачу timestamp, так как futures_exchange_info не требует его
            info = self._retry_api_call(self.client.futures_exchange_info)
        
            # Проверка наличия ожидаемой структуры данных
            if not info or 'symbols' not in info:
                logger.error("Неверная структура ответа от futures_exchange_info")
                return None
            
            for symbol in info['symbols']:
                if symbol['symbol'] == self.symbol:
                    # Находим фильтр LOT_SIZE для минимального количества
                    for f in symbol['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            self.min_qty = float(f['minQty'])
                            logger.info(f"Минимальный размер ордера: {self.min_qty}")
                    return symbol
                
            logger.error(f"Символ {self.symbol} не найден в информации о бирже")
            return None
        
        except Exception as e:
            logger.error(f"Ошибка получения информации о символе: {str(e)}")
            return None

    def _calculate_position_size(self, price: float) -> float:
        """Расчет размера позиции с агрессивным управлением капиталом"""
        try:
            # Получаем актуальные параметры символа
            symbol_info = self._get_symbol_info()
            if not symbol_info:
                logger.error("Не удалось получить информацию о символе")
                return 0.0

            # Рассчитываем базовый размер позиции
            risk_amount = self.current_balance * self.risk_percent
            size = (risk_amount * self.leverage) / price
            
            # Корректируем на волатильность
            atr = self._calculate_atr()
            if math.isnan(atr) or atr <= 0:
                atr = price * 0.01  # Значение по умолчанию 1% от цены
            
            # Проверяем минимальный размер и округляем
            size = max(size, self.min_qty)
            
            # Округляем до шага, допустимого для символа
            step_size = float([f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'][0])
            size = round(size / step_size) * step_size
            
            logger.info(f"Рассчитанный размер позиции: {size} (Цена: {price}, ATR: {atr:.2f}, MinQty: {self.min_qty})")
            
            if size < self.min_qty:
                logger.warning(f"Размер позиции {size} меньше минимального {self.min_qty}")
                return 0.0
                
            return round(size, 6)  # Округляем до 6 знаков для точности
        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции: {str(e)}", exc_info=True)
            return 0.0

    def _calculate_atr(self, period: int = 14) -> float:
        """Расчет Average True Range для управления размером позиции"""
        try:
            high = self.data['high'].values[-period:]
            low = self.data['low'].values[-period:]
            close = self.data['close'].values[-period:]
            return talib.ATR(high, low, close, timeperiod=period)[-1]
        except Exception as e:
            logger.error(f"Ошибка расчета ATR: {str(e)}", exc_info=True)
            return 0.0

    def _prepare_transformer_data(self):
        """Подготовка данных с улучшенной обработкой"""
        try:
            # 1. Проверка колонок
            required_cols = ['open', 'high', 'low', 'close', 'volume',
                            'rsi', 'macd', 'atr', 'obv', 'vwap']
        
            # 2. Расчет индикаторов
            if not self._calculate_indicators():
                raise ValueError("Не удалось рассчитать индикаторы")
            
            # 3. Извлечение фичей
            features = self.data[required_cols].values
        
            # 4. Проверка на некорректные значения
            if np.isnan(features).any() or np.isinf(features).any():
                logger.warning("Обнаружены NaN/Inf, применяем очистку...")
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 5. Раздельная нормализация
            # Цены (0..1)
            price_scaler = MinMaxScaler()
            features[:, :4] = price_scaler.fit_transform(features[:, :4])
        
            # Объемы (RobustScaler)
            volume_scaler = RobustScaler()
            features[:, 4] = volume_scaler.fit_transform(features[:, 4].reshape(-1, 1)).flatten()
        
            # Индикаторы (StandardScaler)
            ind_scaler = StandardScaler()
            features[:, 5:] = ind_scaler.fit_transform(features[:, 5:])
        
            # 6. Формирование последовательностей
            seq_length = 60
            x, y = [], []
        
            for i in range(seq_length, len(features)):
                x.append(features[i-seq_length:i])
                y.append(features[i, 3])  # close price
            
            if len(x) < 10:
                raise ValueError(f"Недостаточно данных: {len(x)} последовательностей")
            
            # В методе (для skaler):
            ind_scaler = StandardScaler()
            features[:, 5:] = ind_scaler.fit_transform(features[:, 5:])
            # Сохраняем scalers
            self.scalers = {
                'price': price_scaler,
                'volume': volume_scaler,
                'indicators': ind_scaler
            }
            
            # Гарантируем правильную 3D структуру [samples, seq_len, features]
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
        
            assert x.ndim == 3, f"X должен быть 3D массивом, получено {x.shape}"
            assert y.ndim == 1, f"Y должен быть 1D массивом, получено {y.shape}"
            assert x.shape[0] == y.shape[0], "Количество samples в X и Y не совпадает"
        
            return x, y, self.scalers
        
        except Exception as e:
            logger.error(f"Ошибка подготовки данных: {str(e)}")
            return None, None, None
      
        
    def _train_transformer_model(self):
        """Обучение Transformer модели

        Returns:
            bool: True если обучение прошло успешно, False в случае ошибки
        """
        try:
            # 1. Подготовка данных
            x, y, scaler = self._prepare_transformer_data()
            if x is None or y is None or scaler is None:
                logger.error("Не удалось подготовить данные для обучения")
                return False
            if x is None or len(x) == 0:
                logger.error("Нет данных для обучения!")
                return False
            # 2. Разделение данных
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42, shuffle=False  # Для временных рядов shuffle=False
            )

            # 3. Конвертация в тензоры с проверкой устройста
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
            y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
            x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
            y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

            # 4. Инициализация модели
            model = TransformerModel(
                input_dim=1,
                model_dim=64,
                num_heads=4,
                num_layers=2,
                output_dim=1
            ).to(device)

            # 5. Настройка обучения
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
            early_stopping = EarlyStopping(patience=5, verbose=True)

            epochs = 20
            best_loss = float('inf')

            # 6. Цикл обучения
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(x_train)
                loss = criterion(outputs.view(-1, 1), y_train_tensor.view(-1, 1))
                loss = criterion(outputs, y_train)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
            
                # Валидация
                model.eval()
                with torch.no_grad():
                    val_outputs = model(x_test)
                    val_loss = criterion(val_outputs, y_test)
            
                scheduler.step(val_loss)
                early_stopping(val_loss)
            
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Эпоха [{epoch+1}/{epochs}] Train Loss: {loss.item():.4f} Val Loss: {val_loss.item():.4f}")
            
                if early_stopping.early_stop:
                    logger.info("Ранняя остановка")
                    break
                
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    best_model = model.state_dict()

            # 7. Сохранение лучшей модели
            model.load_state_dict(best_model)
            self.transformer_model = model.to('cpu')  # Переносим на CPU для сохранения
            self.scaler = scaler
        
            # 8. Оценка качества
            self._evaluate_model(model, x_test, y_test)
            
            # 9. Проверка работы
            logger.info(f"Форма обучающих данных: X={x_train.shape}, Y={y_train.shape}")
            logger.info(f"Пример фичей: {x_train[0][-1]}")  # Последний элемент первой последовательности

            # После обучения:
            if self.transformer_model and len(x_train) > 0:
                feature_names = [
                    'open', 'high', 'low', 'close', 'volume',
                    'rsi', 'macd', 'atr', 'obv', 'vwap'
                ]
                analyze_feature_importance(self.transformer_model, x_train, feature_names)
            logger.info("Transformer модель успешно обучена")
            return True
        
        except Exception as e:
            logger.error(f"Критическая ошибка обучения: {str(e)}", exc_info=True)
            return False

    def train_model(self, epochs=20):
        """Полный цикл обучения модели"""
        try:
            # 1. Загрузка данных (минимум 2 недели)
            if not self.load_historical_data(days=14):
                raise ValueError("Не удалось загрузить исторические данные")
            
            # 2. Проверка достаточности данных
            if len(self.data) < 500:
                raise ValueError(f"Недостаточно данных: {len(self.data)} строк")
            
            # 3. Расчет индикаторов
            self._calculate_indicators()
        
            # 4. Подготовка данных для модели
            x, y, scaler = self._prepare_transformer_data()
            if x is None:
                raise ValueError("Не удалось подготовить данные для обучения")
        
            # 4. Разделение на train/validation
            split = int(0.8 * len(x))
            x_train, x_val = x[:split], x[split:]
            y_train, y_val = y[:split], y[split:]
        
            # 5. Инициализация модели
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = TransformerModel(
                input_dim=10,  # Количество фичей
                model_dim=128,
                num_heads=4,
                num_layers=3,
                output_dim=1
            ).to(device)
        
            # 6. Настройка обучения
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            early_stopping = EarlyStopping(patience=5)
        
            # 7. Цикл обучения
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(torch.tensor(x_train, dtype=torch.float32).to(device))
                loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32).to(device))
                loss.backward()
                optimizer.step()
            
                # Валидация
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(
                        model(torch.tensor(x_val, dtype=torch.float32).to(device)),
                        torch.tensor(y_val, dtype=torch.float32).to(device))
            
                scheduler.step(val_loss)
                early_stopping(val_loss)
            
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
            
                if early_stopping.early_stop:
                    logger.info("Ранняя остановка")
                    break
        
            # 8. Сохранение модели
            self.transformer_model = model.to('cpu')
            self.scaler = scaler
            self.save_model()

            #проверка доп
            if x is None or y is None:
                logger.error("Не удалось подготовить данные для обучения")
                self._send_telegram_alert("⚠️ Ошибка подготовки данных обучения")
                return False
        
            # 9. Визуализация результатов
            self._plot_training_results(x_val, y_val)
        
            logger.info(f"Модель успешно обучена на {len(x)} примерах")
            self._send_telegram_alert(f"✅ Обучение завершено\nLoss: {loss:.4f}\nVal Loss: {val_loss:.4f}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка обучения: {str(e)}")
            self._send_telegram_alert(f"❌ Ошибка обучения: {str(e)[:200]}")
            return False

    def _predict_with_transformer(self, current_price: float):
        """Прогнозирование цены с помощью Transformer

        Args:
            current_price (float): Текущая цена (используется для валидации)
    
        Returns:
            float: Предсказанная цена или None в случае ошибки
        """
        try:
            # 1. Проверка инициализации модели и scaler
            if not hasattr(self, 'transformer_model') or self.transformer_model is None:
                logger.error("Transformer модель не инициализирована")
                return None
            
            if not hasattr(self, 'scaler') or self.scaler is None:
                logger.error("Scaler не инициализирован")
                return None

            # 2. Проверка наличия достаточных данных
            if len(self.data) < 60:
                logger.error(f"Недостаточно данных для прогноза (нужно 60, есть {len(self.data)})")
                return None

            # 3. Подготовка входных данных
            recent_data = self.data['close'].values[-60:].reshape(-1, 1)
        
            # 4. Проверка данных на NaN/Inf
            if np.isnan(recent_data).any() or np.isinf(recent_data).any():
                logger.error("Обнаружены некорректные значения (NaN/Inf) в данных")
                return None

            # 5. Нормализация данных
            try:
                scaled_data = self.scaler.transform(recent_data)
            except ValueError as e:
                logger.error(f"Ошибка нормализации данных: {str(e)}")
                return None

            # 6. Подготовка тензора с учетом устройства
            device = next(self.transformer_model.parameters()).device
            x_input = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

            # 7. Прогнозирование
            self.transformer_model.eval()
            with torch.no_grad():
                try:
                    prediction = self.transformer_model(x_input)
                    prediction_cpu = prediction.cpu()  # Переносим на CPU для inverse_transform
                except RuntimeError as e:
                    logger.error(f"Ошибка во время предсказания: {str(e)}")
                    return None

            # 8. Обратное преобразование и валидация
                try:
                    predicted_price = self.scaler.inverse_transform(prediction_cpu.numpy())[0][0]
            
                    # Проверка разумности предсказания
                    if not (0.1 * current_price < predicted_price < 10 * current_price):
                        logger.warning(f"Странное предсказание: {predicted_price:.2f} при текущей цене {current_price:.2f}")
                        return None
                
                    logger.info(f"Прогноз Transformer: {predicted_price:.2f} (текущая цена: {current_price:.2f})")
                    return predicted_price
            
                except ValueError as e:
                    logger.error(f"Ошибка обратного преобразования: {str(e)}")
                    return None
            
            # 9. Обновление логики прогнозирования
            try:
                # Проверка наличия всех фичей
                feature_columns = [
                    'open', 'high', 'low', 'close', 'volume',
                    'rsi', 'macd', 'atr', 'obv', 'vwap'
                ]
        
                if len(self.data) < 60 or not all(col in self.data.columns for col in feature_columns):
                    logger.error("Недостаточно данных для прогноза")
                    return None
        
                # Подготовка данных
                recent_data = self.data[feature_columns].values[-60:]
                scaled_data = self.scaler.transform(recent_data)
                x_input = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
        
                # Прогноз
                self.transformer_model.eval()
                with torch.no_grad():
                    prediction = self.transformer_model(x_input)
                    predicted_price = self.scaler.inverse_transform(
                        prediction.cpu().numpy().reshape(-1, len(feature_columns))
                    )[0, 3]  # Берем только 'close'
        
                logger.info(f"Прогноз: {predicted_price:.2f} (Текущая цена: {current_price:.2f})")
                return predicted_price
        
            except Exception as e:
                logger.error(f"Ошибка прогноза: {str(e)}", exc_info=True)
                return None

        except Exception as e:
            logger.error(f"Критическая ошибка прогнозирования: {str(e)}", exc_info=True)
            return None

    def _visualize_predictions(self):
        """Визуализация прогнозов Transformer

        Returns:
            BytesIO: Буфер с изображением или None в случае ошибки
        """
        try:
            # 1. Проверка инициализации модели и scaler
            if not hasattr(self, 'transformer_model') or self.transformer_model is None:
                logger.error("Transformer модель не инициализирована")
                return None
            
            if not hasattr(self, 'scaler') or self.scaler is None:
                logger.error("Scaler не инициализирован")
                return None

            # 2. Подготовка данных
            x, y, _ = self._prepare_transformer_data()
            if x is None or y is None:
                logger.error("Не удалось подготовить данные для визуализации")
                return None

            # 3. Проверка достаточности данных
            if len(y) < 60:
                logger.error(f"Недостаточно данных для визуализации (нужно минимум 60, есть {len(y)})")
                return None

            # 4. Прогнозирование
            device = next(self.transformer_model.parameters()).device
            x_test = torch.tensor(x[-60:], dtype=torch.float32).unsqueeze(0).to(device)
        
            self.transformer_model.eval()
            with torch.no_grad():
                predictions = self.transformer_model(x_test).cpu().numpy()
        
            try:
                predictions = self.scaler.inverse_transform(predictions)
            except ValueError as e:
                logger.error(f"Ошибка обратного преобразования: {str(e)}")
                return None

            # 5. Визуализация
            plt.figure(figsize=(14, 7))
        
            # Фактические данные (последние 100 точек)
            actual_data = self.data['close'].values[-100:]
            plt.plot(actual_data, label='Фактические данные', color='blue')
        
            # Прогнозы
            forecast_start = len(actual_data) - 60
            forecast_x = range(forecast_start, forecast_start + len(predictions.flatten()))
            plt.plot(forecast_x, predictions.flatten(), label='Прогноз Transformer', 
                    linestyle='--', color='red', linewidth=2)
        
            plt.legend()
            plt.grid(True)
            plt.title("Сравнение фактических данных и прогноза Transformer")
            plt.xlabel("Временной период")
            plt.ylabel("Цена")
        
            # Сохранение в буфер
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
        
            return buffer

        except Exception as e:
            logger.error(f"Критическая ошибка визуализации: {str(e)}", exc_info=True)
            if 'plt' in locals():
                plt.close()
            return None

    def _plot_training_results(self, x_val, y_val):
        """Графики процесса обучения"""
        try:
            plt.figure(figsize=(15, 5))
    
            # Прогнозы на валидационных данных
            with torch.no_grad():
                predictions = self.transformer_model(
                    torch.tensor(x_val, dtype=torch.float32)
                ).numpy().flatten()
    
            # Денормализация - добавлена проверка наличия inverse_transform
            y_val_orig = y_val  # Значение по умолчанию
            if hasattr(self.scaler, 'inverse_transform'):
                try:
                    y_val_orig = self.scaler.inverse_transform(
                        np.concatenate([x_val[:, -1, :], y_val.reshape(-1, 1)], axis=1)
                    )[:, 3]
                    predictions = self.scaler.inverse_transform(
                        np.concatenate([x_val[:, -1, :], predictions.reshape(-1, 1)], axis=1)
                    )[:, 3]
                except Exception as e:
                    logger.error(f"Ошибка обратного преобразования: {str(e)}")
                    y_val_orig = y_val
    
            # График
            plt.plot(y_val_orig, label='Actual')
            plt.plot(predictions, label='Predicted', alpha=0.7)
            plt.title("Validation Results")
            plt.legend()
    
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close()
    
            self._send_telegram_alert("Результаты обучения", buf)
    
        except Exception as e:
            logger.error(f"Ошибка визуализации: {str(e)}")

    def optimize_hyperparameters(self, n_trials=50):
        study = optuna.create_study(direction='minimize')
        study.optimize(self._objective, n_trials=n_trials)
        self.best_params = study.best_params

    def _calculate_indicators(self):
        """Расчет всех индикаторов с поддержкой TA-Lib, ta и pure Python fallback"""
        try:
            # Проверка наличия базовых колонок
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in self.data.columns for col in required_cols):
                raise ValueError("Отсутствуют базовые колонки OHLCV")

            # Рассчитываем все индикаторы в зависимости от доступности библиотек
            if TALIB_AVAILABLE:
                logger.info("Using TA-Lib for indicators")
                self._calculate_indicators_talib()
            elif TA_AVAILABLE:
                logger.info("Using 'ta' library for indicators")
                self._calculate_indicators_ta()
            else:
                logger.warning("Using pure Python indicator implementations")
                self._calculate_indicators_python()

            # EMA (работает одинаково для всех вариантов)
            self.data['ema5'] = self.data['close'].ewm(span=5, adjust=False).mean()
            self.data['ema10'] = self.data['close'].ewm(span=10, adjust=False).mean()
            
            # Volume indicators
            self.data['volume_ma'] = self.data['volume'].rolling(20).mean()
            self.data['volume_spike'] = (self.data['volume'] > 1.5 * self.data['volume_ma']).astype(int)
            
            # VWAP
            highs = self.data['high'].values
            lows = self.data['low'].values  
            closes = self.data['close'].values
            volumes = self.data['volume'].values
            typical_price = (highs + lows + closes) / 3
            self.data['vwap'] = (typical_price * volumes).cumsum() / volumes.cumsum()
            
            # Заполнение NaN
            self.data = self.data.ffill().bfill().fillna(0)
            
            # Проверка наличия всех необходимых колонок
            required_indicators = ['macd', 'macd_signal', 'rsi', 'stoch_k', 'stoch_d', 
                                'cci', 'bbands_middle', 'adx', 'ema5', 'ema10',
                                'volume_spike', 'vwap']
            
            missing = [ind for ind in required_indicators if ind not in self.data.columns]
            if missing:
                raise ValueError(f"Отсутствуют индикаторы: {missing}")
            
            logger.info("Все индикаторы успешно рассчитаны")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {str(e)}", exc_info=True)
            return False

    def _calculate_indicators_talib(self):
        """Расчет индикаторов через TA-Lib"""
        closes = self.data['close'].ffill().values
        highs = self.data['high'].ffill().values
        lows = self.data['low'].ffill().values
        volumes = self.data['volume'].fillna(0).values

        # MACD
        macd, macd_signal, _ = talib.MACD(closes)
        self.data['macd'] = macd
        self.data['macd_signal'] = macd_signal
            
        # Другие индикаторы
        self.data['rsi'] = talib.RSI(closes, timeperiod=14)
        self.data['atr'] = talib.ATR(highs, lows, closes, timeperiod=14)
        self.data['obv'] = talib.OBV(closes, volumes)
            
        # Stochastic
        slowk, slowd = talib.STOCH(highs, lows, closes,
                                fastk_period=5, slowk_period=3,
                                slowd_period=3, slowk_matype=0, slowd_matype=0)
        self.data['stoch_k'] = slowk
        self.data['stoch_d'] = slowd
            
        # Остальные индикаторы
        self.data['cci'] = talib.CCI(highs, lows, closes, timeperiod=14)
        upper, middle, lower = talib.BBANDS(closes, timeperiod=20)
        self.data['bbands_middle'] = middle
        self.data['adx'] = talib.ADX(highs, lows, closes, timeperiod=14)

    def _calculate_indicators_ta(self):
        """Расчет индикаторов через библиотеку ta"""
        import ta
        
        df = self.data.copy()
        
        # MACD
        macd_indicator = ta.trend.MACD(df['close'])
        self.data['macd'] = macd_indicator.macd()
        self.data['macd_signal'] = macd_indicator.macd_signal()
        
        # RSI
        self.data['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # ATR
        self.data['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # OBV
        self.data['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        self.data['stoch_k'] = stoch.stoch()
        self.data['stoch_d'] = stoch.stoch_signal()
        
        # CCI
        self.data['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        self.data['bbands_middle'] = bollinger.bollinger_mavg()
        
        # ADX
        self.data['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()

    def _calculate_indicators_python(self):
        """Pure Python реализации индикаторов"""
        closes = self.data['close'].ffill().values
        highs = self.data['high'].ffill().values
        lows = self.data['low'].ffill().values
        volumes = self.data['volume'].fillna(0).values
        
        # MACD (Python реализация)
        def calculate_macd_python(prices, fast=12, slow=26, signal=9):
            ema_fast = pd.Series(prices).ewm(span=fast).mean()
            ema_slow = pd.Series(prices).ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            return macd, signal_line, macd - signal_line
        
        macd, macd_signal, _ = calculate_macd_python(closes)
        self.data['macd'] = macd
        self.data['macd_signal'] = macd_signal
        
        # RSI (Python реализация)
        def calculate_rsi_python(prices, period=14):
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = pd.Series(gains).rolling(period).mean()
            avg_losses = pd.Series(losses).rolling(period).mean()
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        self.data['rsi'] = calculate_rsi_python(closes)
        
        # ATR (Python реализация)
        def calculate_atr_python(high, low, close, period=14):
            tr = np.maximum(high - low, 
                        np.maximum(np.abs(high - close), 
                                    np.abs(low - close)))
            atr = pd.Series(tr).rolling(period).mean()
            return atr
        
        self.data['atr'] = calculate_atr_python(highs, lows, closes)
        
        # OBV (Python реализация)
        def calculate_obv_python(close, volume):
            obv = [0]
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv.append(obv[-1] + volume[i])
                elif close[i] < close[i-1]:
                    obv.append(obv[-1] - volume[i])
                else:
                    obv.append(obv[-1])
            return obv
        
        self.data['obv'] = calculate_obv_python(closes, volumes)
        
        # Stochastic (упрощенная Python реализация)
        def calculate_stoch_python(high, low, close, k_period=14, d_period=3):
            lowest_low = pd.Series(low).rolling(k_period).min()
            highest_high = pd.Series(high).rolling(k_period).max()
            
            stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            stoch_d = stoch_k.rolling(d_period).mean()
            return stoch_k, stoch_d
        
        stoch_k, stoch_d = calculate_stoch_python(highs, lows, closes)
        self.data['stoch_k'] = stoch_k
        self.data['stoch_d'] = stoch_d
        
        # CCI (упрощенная реализация)
        self.data['cci'] = 0  # Упрощенно
        
        # Bollinger Bands (Python реализация)
        def calculate_bbands_python(prices, period=20, std=2):
            sma = pd.Series(prices).rolling(period).mean()
            rolling_std = pd.Series(prices).rolling(period).std()
            upper = sma + (rolling_std * std)
            lower = sma - (rolling_std * std)
            return sma, upper, lower
        
        bbands_middle, _, _ = calculate_bbands_python(closes)
        self.data['bbands_middle'] = bbands_middle
        
        # ADX (упрощенная реализация)
        self.data['adx'] = 0  # Упрощенно

    def _calculate_extended_indicators(self):
        """Расчёт дополнительных индикаторов"""
        closes = self.data['close'].values
        # Добавьте новые индикаторы:
        self.data['volume_ma'] = self.data['volume'].rolling(20).mean()
        self.data['price_trend'] = talib.LINEARREG(closes, timeperiod=14)

    def _check_entry_signal(self, indicators: Dict[str, float]) -> bool:
        """Проверка условий для входа в агрессивной стратегии
        
        Args:
            indicators: Словарь с рассчитанными индикаторами
        
        Returns:
            bool: True если есть сильный сигнал на вход, False в противном случае
        """
        if not isinstance(indicators, dict):
            logger.error(f"Некорректный тип индикаторов: {type(indicators)}")
            return False
        """Проверка условий для входа в агрессивной стратегии"""
        try:
            # Проверяем наличие всех необходимых индикаторов
            required_keys = ['ema5', 'ema10', 'rsi', 'macd', 'macd_signal',
                            'stoch_k', 'stoch_d', 'volume_spike', 'vwap',
                            'cci', 'bbands_middle', 'adx', 'close']
        
            # Проверяем наличие всех ключей
            missing_keys = [key for key in required_keys if key not in indicators]
            if missing_keys:
                logger.error(f"Отсутствуют индикаторы: {missing_keys}")
                return False
            
            # Проверяем NaN значения
            for key, value in indicators.items():
                if isinstance(value, (float, int)) and np.isnan(value):
                    logger.warning(f"NaN значение в индикаторе {key}")
                    return False
        
            # Условия для LONG
            long_conditions = [
                indicators['ema5'] > indicators['ema10'],
                45 < indicators['rsi'] < 70,
                indicators['macd'] > indicators['macd_signal'],  # Теперь этот ключ точно есть
                indicators['stoch_k'] > indicators['stoch_d'],
                indicators['volume_spike'] == 1,
                indicators['close'] > indicators['vwap'],
                indicators['cci'] > -100,
                indicators['close'] > indicators['bbands_middle'],
                indicators['adx'] > 20
            ]
        
            # Условия для SHORT
            short_conditions = [
                indicators['ema5'] < indicators['ema10'],
                30 < indicators['rsi'] < 55,
                indicators['macd'] < indicators['macd_signal'],
                indicators['stoch_k'] < indicators['stoch_d'],
                indicators['volume_spike'] == 1,
                indicators['close'] < indicators['vwap'],
                indicators['cci'] < 100,
                indicators['close'] < indicators['bbands_middle'],
                indicators['adx'] > 20
            ]
        
            # Подсчет условий
            long_count = sum(long_conditions)
            short_count = sum(short_conditions)
        
            # Проверка сигналов
            threshold = 6  # Минимум 6 из 9 условий
            if long_count >= threshold:
                logger.info(f"LONG сигнал ({long_count}/9 условий)")
                return True
            elif short_count >= threshold:
                logger.info(f"SHORT сигнал ({short_count}/9 условий)")
                return True
            
            

            if not indicators or not isinstance(indicators, dict):
                logger.warning("Некорректные индикаторы для проверки сигнала")
                return False

            # Проверка наличия всех необходимых ключей
            required_keys = ['ema5', 'ema10', 'rsi', 'macd', 'signal', 
                            'stoch_k', 'stoch_d', 'volume_spike', 
                            'vwap', 'cci', 'bbands_middle', 'adx', 'close']
        
            # Если каких-то индикаторов нет, пытаемся их рассчитать
            for key in required_keys:
                if key not in indicators:
                    if key in self.data.columns:
                        indicators[key] = self.data[key].iloc[-1]
                    else:
                        logger.error(f"Не удалось получить индикатор: {key}")
                        return False
            
            # Добавляем EMA расчеты если их нет
            if 'ema5' not in indicators:
                indicators['ema5'] = self.data['close'].ewm(span=5).mean().iloc[-1]
            if 'ema10' not in indicators:
                indicators['ema10'] = self.data['close'].ewm(span=10).mean().iloc[-1]
        
            missing_keys = [key for key in required_keys if key not in indicators]
            if missing_keys:
                logger.error(f"Отсутствуют ключи индикаторов: {missing_keys}")
                return False

            # Проверка на NaN значения
            nan_values = [key for key, value in indicators.items() 
                         if isinstance(value, float) and np.isnan(value)]
            if nan_values:
                logger.warning(f"Обнаружены NaN значения в индикаторах: {nan_values}")
                return False
            
            if not indicators or len(indicators) == 0:
                logger.warning("Пустой словарь индикаторов")
                return False
            # Добавлена проверка типа indicators
            if not isinstance(indicators, dict):
                logger.error(f"Некорректный тип индикаторов: {type(indicators)}")
                return False
            
            if not indicators:
                logger.warning("Пустой словарь индикаторов")
                return False
            # Проверка наличия всех необходимых ключей
            required_keys = ['ema5', 'ema10', 'rsi', 'macd', 'signal', 'stoch_k', 
                           'stoch_d', 'volume_spike', 'vwap', 'cci', 'bbands_middle', 'adx', 'close']
            missing_keys = [key for key in required_keys if key not in indicators]
            if missing_keys:
                logger.error(f"Отсутствуют ключи индикаторов: {missing_keys}")
                return False

            # Проверка на NaN значения
            nan_values = [k for k, v in indicators.items() if isinstance(v, (int, float)) and np.isnan(v)]
            if nan_values:
                logger.warning(f"Обнаружены NaN значения в индикаторах: {nan_values}")
                return False

            # Условия для лонга
            long_conditions = [
                indicators['ema5'] > indicators['ema10'],
                45 < indicators['rsi'] < 70,
                indicators['macd'] > indicators['macd_signal'],
                indicators['stoch_k'] > indicators['stoch_d'],
                indicators['volume_spike'] == 1,
                indicators['close'] > indicators['vwap'],
                indicators['cci'] > -100,
                indicators['close'] > indicators['bbands_middle'],
                indicators['adx'] > 20
            ]
    
            # Условия для шорта
            short_conditions = [
                indicators['ema5'] < indicators['ema10'],
                30 < indicators['rsi'] < 55,
                indicators['macd'] < indicators['macd_signal'],
                indicators['stoch_k'] < indicators['stoch_d'],
                indicators['volume_spike'] == 1,
                indicators['close'] < indicators['vwap'],
                indicators['cci'] < 100,
                indicators['close'] < indicators['bbands_middle'],
                indicators['adx'] > 20
            ]
    
            # Подсчет выполненных условий
            long_count = sum(long_conditions)
            short_count = sum(short_conditions)
    
            # Требуется большинство условий (6 из 9)
            threshold = 6
            long_signal = long_count >= threshold
            short_signal = short_count >= threshold
    
            if long_signal or short_signal:
                signal_type = 'LONG' if long_signal else 'SHORT'
                logger.info(f"Сигнал на вход: {signal_type} (условий: {long_count if long_signal else short_count}/9)")
                return True
        
            return False

        except Exception as e:
            logger.error(f"Ошибка проверки сигнала: {str(e)}", exc_info=True)
            return False

    def _check_exit_signal(self, indicators: Dict[str, float]) -> bool:
        """Проверка условий для выхода из позиции

        Args:
            indicators: Словарь с рассчитанными индикаторами
        
        Returns:
            bool: True если нужно выйти из позиции, False в противном случае
        """
        try:
            
            # Проверка наличия открытой позиции
            if not self.current_position or not isinstance(self.current_position, dict):
                logger.debug("Нет активной позиции для проверки выхода")
                return False

            # Проверка обязательных полей в позиции
            required_position_fields = ['side', 'entry_price']
            if any(field not in self.current_position for field in required_position_fields):
                logger.error(f"В current_position отсутствуют обязательные поля: {required_position_fields}")
                return False

            # Проверка наличия необходимых индикаторов
            required_indicators = ['close', 'rsi', 'ema5', 'ema10', 'macd', 'signal', 'bbands_middle']
            if not indicators or any(ind not in indicators for ind in required_indicators):
                missing = [ind for ind in required_indicators if ind not in indicators]
                logger.error(f"Отсутствуют необходимые индикаторы: {missing}")
                return False

            current_price = indicators['close']
            entry_price = self.current_position['entry_price']
            position_side = self.current_position['side']

            # Валидация цен
            if not all(isinstance(p, (int, float)) for p in [current_price, entry_price]):
                logger.error("Некорректные значения цен (не числовые)")
                return False

            exit_conditions = []
        
            if position_side == 'BUY':
                take_profit = entry_price * (1 + self.take_profit)
                stop_loss = entry_price * (1 - self.stop_loss)
            
                exit_conditions = [
                    current_price >= take_profit,
                    current_price <= stop_loss,
                    indicators['rsi'] > 70,
                    indicators['ema5'] < indicators['ema10'],
                    indicators['macd'] < indicators['signal'],
                    indicators['close'] < indicators['bbands_middle']
                ]
            
                # Логирование условий выхода для лонга
                if any(exit_conditions):
                    condition_names = [
                        "Take Profit", "Stop Loss", "RSI > 70", 
                        "EMA5 < EMA10", "MACD < Signal", "Price < BB Middle"
                    ]
                    triggered = [name for name, cond in zip(condition_names, exit_conditions) if cond]
                    logger.info(f"Условия выхода из лонга: {', '.join(triggered)}")

            elif position_side == 'SELL':
                take_profit = entry_price * (1 - self.take_profit)
                stop_loss = entry_price * (1 + self.stop_loss)
            
                exit_conditions = [
                    current_price <= take_profit,
                    current_price >= stop_loss,
                    indicators['rsi'] < 30,
                    indicators['ema5'] > indicators['ema10'],
                    indicators['macd'] > indicators['signal'],
                    indicators['close'] > indicators['bbands_middle']
                ]
            
                # Логирование условий выхода для шорта
                if any(exit_conditions):
                    condition_names = [
                        "Take Profit", "Stop Loss", "RSI < 30", 
                        "EMA5 > EMA10", "MACD > Signal", "Price > BB Middle"
                    ]
                    triggered = [name for name, cond in zip(condition_names, exit_conditions) if cond]
                    logger.info(f"Условия выхода из шорта: {', '.join(triggered)}")

            else:
                logger.error(f"Неизвестное направление позиции: {position_side}")
                return False

            return any(exit_conditions)

        except Exception as e:
            logger.error(f"Ошибка при проверке условий выхода: {str(e)}", exc_info=True)
            return False

    def _handle_websocket_message(self, msg: Dict[str, Any]):
        """Обработчик сообщений WebSocket с улучшенной обработкой ошибок и логированием"""
        try:
            # Проверка структуры сообщения
            if not isinstance(msg, dict) or 'e' not in msg or 'k' not in msg:
                logger.warning(f"Некорректная структура сообщения: {msg.keys()}")
                return

            # Обработка только закрытых свечей (kline)
            if msg['e'] == 'kline' and msg['k']['x']:
                kline = msg['k']

                # Проверка обязательных полей свечи
                required_kline_fields = ['t', 'o', 'h', 'l', 'c', 'v']
                missing = [field for field in required_kline_fields if field not in kline]
                if missing:
                    logger.error(f"Отсутствуют поля в свече: {missing}")
                    return

                try:
                    current_time = kline['t']
                    current_price = float(kline['c'])

                    # Проверка на дубликаты свечей
                    if hasattr(self, 'last_candle_time') and self.last_candle_time == current_time:
                        logger.debug(f"Пропуск дубликата свечи: {current_time}")
                        return

                    self.last_candle_time = current_time
                    logger.info(f"Новая свеча: {current_time} | Цена: {current_price:.4f}")

                    # Создание новой свечи с проверкой значений
                    new_candle = {
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': current_price,
                        'volume': float(kline['v'])
                    }

                    # Проверка на корректность цен
                    if not (new_candle['low'] <= new_candle['close'] <= new_candle['high'] and 
                            new_candle['low'] <= new_candle['open'] <= new_candle['high']):
                        logger.error(f"Некорректные цены в свече: {new_candle}")
                        return

                    # Обновление данных
                    self.data = pd.concat([
                        self.data, 
                        pd.DataFrame([new_candle])
                    ], ignore_index=True).tail(self.data_window_size)

                    # Расчет индикаторов
                    if not self._calculate_indicators():
                        logger.warning("Не удалось рассчитать индикаторы")
                        return

                    # Создаем словарь индикаторов
                    indicators = {
                        'close': self.data['close'].iloc[-1],
                        'ema5': self.data['ema5'].iloc[-1],
                        'ema10': self.data['ema10'].iloc[-1],
                        'rsi': self.data['rsi'].iloc[-1],
                        'macd': self.data['macd'].iloc[-1],
                        'macd_signal': self.data['macd_signal'].iloc[-1],
                        'stoch_k': self.data['stoch_k'].iloc[-1],
                        'stoch_d': self.data['stoch_d'].iloc[-1],
                        'volume_spike': self.data['volume_spike'].iloc[-1],
                        'vwap': self.data['vwap'].iloc[-1],
                        'cci': self.data['cci'].iloc[-1],
                        'bbands_middle': self.data['bbands_middle'].iloc[-1],
                        'adx': self.data['adx'].iloc[-1]
                    }

                    # Проверка сигналов
                    if not self.current_position:
                        if self._check_entry_signal(indicators):
                            side = 'BUY' if indicators['ema5'] > indicators['ema10'] else 'SELL'
                            logger.info(f"Сигнал на открытие {side} позиции")
                            self._open_position(current_price, side)
                    else:
                        if self._check_exit_signal(indicators):
                            logger.info("Сигнал на закрытие позиции")
                            self._close_position(current_price)

                    # Отправка live-графика каждые 30 свечей
                    if len(self.data) % 30 == 0:
                        chart = self._generate_chart()
                        if chart:
                            self._send_telegram_alert(
                                f"Обновление {self.symbol}",
                                chart=chart
                            )

                    # Переобучение каждые 24 часа
                    if not hasattr(self, 'last_training_time') or \
                       (datetime.now() - self.last_training_time).total_seconds() > 86400:
                        logger.info("Запуск периодического переобучения...")
                        if self.train_model(epochs=5):
                            self.last_training_time = datetime.now()
                            self._send_telegram_alert("🔄 Модель успешно переобучена")

                    # Оптимизация раз в день
                    if datetime.now().hour == 0 and len(self.data) % 144 == 0:
                        optimizer = HyperparameterOptimizer(self)
                        optimizer.optimize(n_trials=30)

                    # Еженедельное обучение
                    if datetime.now().weekday() == 0:
                        if not hasattr(self, 'last_training') or \
                           (datetime.now() - self.last_training).days >= 7:
                            self.train_model()
                            self.last_training = datetime.now()

                except (ValueError, TypeError) as e:
                    logger.error(f"Ошибка преобразования данных свечи: {str(e)}")
                    return

        except Exception as e:
            error_msg = f"Критическая ошибка обработки сообщения: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._send_telegram_alert(f"⚠️ Ошибка WS: {error_msg[:200]}")

    def _get_max_position_size(self) -> float:
        """Рассчитывает максимально допустимый размер позиции на основе баланса и плеча
    
        Returns:
            float: Максимальный размер позиции в базовой валюте (например, BTC для BTCUSDT)
        """
        try:
            # Получаем текущий баланс
            if not self._update_account_info():
                logger.error("Не удалось обновить информацию об аккаунте")
                return 0.0
            
            # Рассчитываем максимальный размер с учетом плеча
            max_size = (self.current_balance * self.leverage) / self.data['close'].iloc[-1]
        
            # Получаем информацию о символе для проверки ограничений
            symbol_info = self._get_symbol_info()
            if symbol_info:
                for f in symbol_info['filters']:
                    if f['filterType'] == 'MARKET_LOT_SIZE':
                        max_market_size = float(f['maxQty'])
                        max_size = min(max_size, max_market_size)
                    
            logger.debug(f"Максимальный размер позиции: {max_size:.6f}")
            return max_size
        
        except Exception as e:
            logger.error(f"Ошибка расчета максимального размера позиции: {str(e)}")
            return 0.0
    def _adjust_quantity(self, quantity: float) -> float:
        """Корректирует количество согласно правилам биржи
    
        Args:
            quantity: Исходное количество
        
        Returns:
            float: Скорректированное количество
        """
        try:
            symbol_info = self._get_symbol_info()
            if not symbol_info:
                return quantity
            
            # Находим шаг размера лота
            for f in symbol_info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = float(f['stepSize'])
                    # Округляем до ближайшего допустимого шага
                    adjusted = round(quantity / step_size) * step_size
                    logger.debug(f"Корректировка количества: {quantity} -> {adjusted}")
                    return adjusted
                
            return quantity
        except Exception as e:
            logger.error(f"Ошибка корректировки количества: {str(e)}")
            return quantity

    def _get_price_precision(self) -> int:
        """Возвращает количество знаков после запятой для цены
    
        Returns:
            int: Количество знаков после запятой
        """
        try:
            symbol_info = self._get_symbol_info()
            if symbol_info:
                return symbol_info['pricePrecision']
            return 2  # Значение по умолчанию
        except Exception as e:
            logger.error(f"Ошибка получения точности цены: {str(e)}")
            return 2    

    def _cancel_order(self, order_id: int) -> bool:
        """Отменяет ордер по его ID
    
        Args:
            order_id: ID ордера для отмены
        
        Returns:
            bool: True если ордер успешно отменен, False в случае ошибки
        """
        try:
            result = self._retry_api_call(
                self.client.futures_cancel_order,
                symbol=self.symbol,
                orderId=order_id
            )
            return result.get('status') == 'CANCELED'
        except Exception as e:
            logger.error(f"Ошибка отмены ордера {order_id}: {str(e)}")
            return False
    
    def _open_position(self, price: float, side: str) -> bool:
        """Открытие позиции с агрессивными параметрами

        Args:
            price: Цена входа
            side: Направление позиции ('BUY' или 'SELL')

        Returns:
            bool: True если позиция успешно открыта, False в случае ошибки
        """
        try:
            # Валидация входных параметров
            if not isinstance(price, (int, float)) or price <= 0:
                logger.error(f"Некорректная цена: {price}")
                return False
            
            if side not in ['BUY', 'SELL']:
                logger.error(f"Некорректное направление позиции: {side}")
                return False

            # Расчет размера позиции
            size = self._calculate_position_size(price)
            max_size = self._get_max_position_size()
            if size < self.min_qty or math.isnan(size):
                logger.warning(f"Слишком маленький размер позиции: {size}")
                return 0.0
            
            logger.info(f"Попытка открытия {side} позиции: {size:.6f} {self.symbol} по цене {price:.2f}")

            # Размещение рыночного ордера
            try:
                order = self._retry_api_call(
                    self.client.futures_create_order,
                    symbol=self.symbol,
                    side=side,
                    type='MARKET',
                    quantity=self._adjust_quantity(size)
                )
            
                if not order or 'orderId' not in order:
                    logger.error("Не удалось разместить рыночный ордер")
                    return False
            except Exception as e:
                logger.error(f"Ошибка размещения рыночного ордера: {str(e)}")
                return False

            # Расчет TP/SL с округлением до шага цены
            price_precision = self._get_price_precision()
            if side == 'BUY':
                take_profit = round(price * (1 + self.take_profit), price_precision)
                stop_loss = round(price * (1 - self.stop_loss), price_precision)
            else:
                take_profit = round(price * (1 - self.take_profit), price_precision)
                stop_loss = round(price * (1 + self.stop_loss), price_precision)

            # Размещение TP/SL ордеров
            try:
                # Take Profit
                self._retry_api_call(
                    self.client.futures_create_order,
                    symbol=self.symbol,
                    side='SELL' if side == 'BUY' else 'BUY',
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=take_profit,
                    closePosition=True,
                    quantity=self._adjust_quantity(size)
                )
            
                # Stop Loss
                self._retry_api_call(
                    self.client.futures_create_order,
                    symbol=self.symbol,
                    side='SELL' if side == 'BUY' else 'BUY',
                    type='STOP_MARKET',
                    stopPrice=stop_loss,
                    closePosition=True,
                    quantity=self._adjust_quantity(size)
                )
            except Exception as e:
                logger.error(f"Ошибка размещения TP/SL ордеров: {str(e)}")
                # Отменяем основную позицию при ошибке
                self._cancel_order(order['orderId'])
                return False

            # Обновление состояния позиции
            self.current_position = {
                'side': side,
                'entry_price': price,
                'size': size,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'order_id': order['orderId'],
                'opened_at': datetime.utcnow(),  # Используем UTC для консистентности
                'tp_order_id': None,  # Можно добавить реальные ID ордеров TP/SL
                'sl_order_id': None
            }
        
            self.trade_count += 1
            if not self._update_account_info():
                logger.warning("Не удалось обновить информацию об аккаунте")

            # Получение прогноза
            prediction = self._predict_with_transformer(price)
            prediction_text = f" | Прогноз: {prediction:.2f}" if prediction is not None else ""

            # Формирование сообщения
            message = (
                f"🚀 <b>Открыта позиция #{self.trade_count}</b>\n"
                f"┌ Пара: {self.symbol}\n"
                f"├ Тип: {side}\n"
                f"├ Размер: {size:.6f}\n"
                f"├ Цена: {price:.4f}{prediction_text}\n"
                f"├ Плечо: {self.leverage}x\n"
                f"├ TP: {take_profit:.4f} (+{self.take_profit*100:.1f}%)\n"
                f"├ SL: {stop_loss:.4f} (-{self.stop_loss*100:.1f}%)\n"
                f"└ Баланс: {self.current_balance:.2f} USDT"
            )

            # Отправка уведомления
            chart = self._visualize_predictions()
            self._send_telegram_alert(message, chart)
        
            logger.info(f"Позиция успешно открыта: {side} {size:.6f} {self.symbol}")
            return True

        except Exception as e:
            logger.error(f"Критическая ошибка при открытии позиции: {str(e)}", exc_info=True)
            self._send_telegram_alert(f"❌ Ошибка открытия позиции: {str(e)[:200]}...")  # Обрезаем длинные сообщения
            return False

    def _close_position(self, price: float) -> bool:
        """Закрытие позиции с расчетом статистики

        Args:
            price: Цена закрытия

        Returns:
            bool: True если позиция успешно закрыта, False в случае ошибки
        """
        try:
            # Проверка наличия открытой позиции
            if not self.current_position or not isinstance(self.current_position, dict):
                logger.warning("Нет активной позиции для закрытия")
                return False

            # Проверка обязательных полей в позиции
            required_fields = ['side', 'size', 'entry_price', 'opened_at']
            missing_fields = [field for field in required_fields if field not in self.current_position]
            if missing_fields:
                logger.error(f"В позиции отсутствуют обязательные поля: {missing_fields}")
                return False

            # Определение направления закрытия
            position_side = self.current_position['side']
            if position_side not in ['BUY', 'SELL']:
                logger.error(f"Некорректное направление позиции: {position_side}")
                return False

            close_side = 'SELL' if position_side == 'BUY' else 'BUY'
            size = self.current_position['size']

            # Проверка размера позиции
            if size <= 0 or not isinstance(size, (int, float)):
                logger.error(f"Некорректный размер позиции: {size}")
                return False

            logger.info(f"Попытка закрытия позиции {position_side} {size:.6f} {self.symbol} по цене {price:.4f}")

            # Размещение рыночного ордера на закрытие
            try:
                order = self._retry_api_call(
                    self.client.futures_create_order,
                    symbol=self.symbol,
                    side=close_side,
                    type='MARKET',
                    quantity=self._adjust_quantity(size)
                )
            
                if not order or 'orderId' not in order:
                    logger.error("Не удалось разместить ордер на закрытие")
                    return False
            except Exception as e:
                logger.error(f"Ошибка размещения ордера на закрытие: {str(e)}")
                return False

            # Отмена всех активных ордеров (TP/SL)
            if not self._cancel_all_orders():
                logger.warning("Не удалось отменить все ордера")

            # Расчет PnL с учетом комиссий
            entry_price = self.current_position['entry_price']
            if position_side == 'BUY':
                pnl = (price - entry_price) * size
                pnl_percent = (price - entry_price) / entry_price * 100 * self.leverage
            else:
                pnl = (entry_price - price) * size
                pnl_percent = (entry_price - price) / entry_price * 100 * self.leverage

            # Обновление статистики
            self.profit_loss += pnl
            if pnl > 0:
                self.win_count += 1
        
            # Расчет win rate с защитой от деления на ноль
            self.win_rate = (self.win_count / self.trade_count) * 100 if self.trade_count > 0 else 0

            # Расчет длительности позиции
            duration = datetime.utcnow() - self.current_position['opened_at']
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, _ = divmod(remainder, 60)

            # Формирование сообщения
            message = (
                f"🔴 <b>Закрыта позиция #{self.trade_count}</b>\n"
                f"┌ Пара: {self.symbol}\n"
                f"├ Тип: {position_side}\n"
                f"├ Размер: {size:.6f}\n"
                f"├ Цена входа: {entry_price:.4f}\n"
                f"├ Цена выхода: {price:.4f}\n"
                f"├ Длительность: {int(hours)}ч {int(minutes)}м\n"
                f"├ PnL: {pnl:+.2f} USDT ({pnl_percent:+.2f}%)\n"
                f"├ Win Rate: {self.win_rate:.1f}%\n"
                f"└ Общий PnL: {self.profit_loss:+.2f} USDT"
            )

            # Отправка уведомления
            self._send_telegram_alert(message)

            # Очистка текущей позиции
            self.current_position = None

            # Обновление информации об аккаунте
            if not self._update_account_info():
                logger.warning("Не удалось обновить информацию об аккаунте")

            logger.info(f"Позиция успешно закрыта. PnL: {pnl:+.2f} USDT")
            return True

        except Exception as e:
            error_msg = f"Критическая ошибка при закрытии позиции: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._send_telegram_alert(f"❌ Ошибка закрытия: {error_msg[:200]}...")
            return False

    def _cancel_all_orders(self) -> bool:
        """Отмена всех активных ордеров для текущего символа

        Returns:
            bool: True если все ордера успешно отменены или их нет, False в случае ошибки
        """
        try:
            # Получаем список активных ордеров
            orders = self._retry_api_call(
                self.client.futures_get_open_orders,
                symbol=self.symbol
            )
        
            # Проверка ответа от API
            if not isinstance(orders, list):
                logger.error(f"Некорректный формат списка ордеров: {type(orders)}")
                return False

            # Если нет ордеров - возвращаем True
            if not orders:
                logger.debug(f"Нет активных ордеров для {self.symbol}")
                return True

            success_count = 0
            error_count = 0
        
            # Отменяем каждый ордер
            for order in orders:
                try:
                    # Проверка наличия orderId
                    if 'orderId' not in order:
                        logger.warning(f"Ордер без orderId: {order}")
                        error_count += 1
                        continue
                    
                    result = self._retry_api_call(
                        self.client.futures_cancel_order,
                        symbol=self.symbol,
                        orderId=order['orderId']
                    )
                
                    # Проверка успешности отмены
                    if isinstance(result, dict) and result.get('status') == 'CANCELED':
                        success_count += 1
                        logger.debug(f"Ордер {order['orderId']} успешно отменен")
                    else:
                        error_count += 1
                        logger.warning(f"Не удалось отменить ордер {order['orderId']}")
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Ошибка при отмене ордера {order.get('orderId', 'unknown')}: {str(e)}")

            # Логирование итогов
            if error_count == 0:
                logger.info(f"Все {success_count} активных ордера для {self.symbol} успешно отменены")
                return True
            else:
                logger.warning(f"Отменено {success_count} ордеров, не удалось отменить {error_count}")
                return False

        except Exception as e:
            logger.error(f"Критическая ошибка при отмене ордеров: {str(e)}", exc_info=True)
            return False
        
    def save_model(self, path='transformer_model.pth'):
        """Упрощенное сохранение модели"""
        try:
            if not hasattr(self, 'transformer_model') or self.transformer_model is None:
                logger.error("Модель не инициализирована для сохранения")
                return False

            # Убираем создание директории для простого пути
            torch.save({
                'model_state_dict': self.transformer_model.state_dict(),
                'scaler_mean': getattr(self.scaler, 'mean_', None),
                'scaler_scale': getattr(self.scaler, 'scale_', None)
            }, path)
            
            logger.info(f"Модель успешно сохранена в {path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {str(e)}", exc_info=True)
            return False
    
    def load_model(self, path='transformer_model.pth'):
        """Загружает модель с проверкой архитектуры"""
        try:
            if not os.path.exists(path):
                logger.warning(f"Файл модели {path} не найден")
                return self._init_new_model()
        
            # Сначала создаем модель с параметрами по умолчанию
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._init_new_model()  # Инициализируем новую модель
        
            # Загружаем checkpoint
            checkpoint = torch.load(path, map_location=device)
        
            # Проверяем структуру checkpoint
            if not all(k in checkpoint for k in ['model_state_dict', 'model_config', 'scaler_mean', 'scaler_scale']):
                logger.error("Неверная структура файла модели")
                return self._init_new_model()
        
            # Сравниваем конфигурации
            current_config = {
                'num_layers': self.transformer_model.num_layers,
                'model_dim': self.transformer_model.model_dim,
                'num_heads': self.transformer_model.num_heads
            }
        
            if checkpoint['model_config'] != current_config:
                logger.warning(f"Конфигурация модели не совпадает. Создаю новую модель.")
                logger.info(f"Загруженная конфигурация: {checkpoint['model_config']}")
                logger.info(f"Текущая конфигурация: {current_config}")
                return self._init_new_model()
        
            # Загружаем веса с обработкой несоответствий
            model_state_dict = checkpoint['model_state_dict']
            current_state_dict = self.transformer_model.state_dict()
        
            # Фильтруем только совпадающие ключи
            filtered_state_dict = {k: v for k, v in model_state_dict.items() 
                                 if k in current_state_dict and v.size() == current_state_dict[k].size()}
        
            # Обновляем state_dict
            current_state_dict.update(filtered_state_dict)
            self.transformer_model.load_state_dict(current_state_dict)
            self.transformer_model.to(device)
        
            # Загружаем scaler
            self.scaler.mean_ = checkpoint['scaler_mean']
            self.scaler.scale_ = checkpoint['scaler_scale']
        
            logger.info(f"Модель успешно загружена с {device}")
            return True
        
        except Exception as e:
            logger.error(f"Критическая ошибка загрузки модели: {str(e)}", exc_info=True)
            return self._init_new_model()
    
    def start(self):
        """Запуск бота с проверкой соединения"""
        try:
            # Проверка версии библиотеки
            self._check_binance_version()           
            # Явная проверка модели перед запуском
            model_path = 'transformer_model.pth'
            if not os.path.exists(model_path):
                logger.warning("Файл модели не найден, инициализирую новую...")
                if not hasattr(self, 'transformer_model') or self.transformer_model is None:
                    self._init_new_model()
                self.save_model()  # Сразу сохраняем новую модель
                logger.info(f"Новая модель сохранена в {os.path.abspath(model_path)}")

            # Проверка соединения
            if not self._check_api_connection():
                self._send_telegram_alert("❌ Ошибка соединения с Binance API")
                return False
        
            if not self.check_connection():
                logger.error("Не удалось подключиться к Binance API")
                self._send_telegram_alert("❌ Критическая ошибка: проверьте соединение и API-ключи")
                return False

            # Попытки подключения WebSocket
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                if self.start_websocket():
                    break
                
                if attempt < max_attempts:
                    logger.warning(f"Попытка {attempt} из {max_attempts} не удалась. Повтор через 5 сек...")
                    time.sleep(5)
            else:
                raise ConnectionError(f"Не удалось подключиться после {max_attempts} попыток")
        
            # Проверка соединения
            if not self.is_websocket_connected():
                raise ConnectionError("WebSocket не активен после подключения")
            
            logger.info("Бот успешно запущен")
            return True

            # Настройка плеча и информации об аккаунте
            if not self._setup_leverage() or not self._update_account_info():
                return False

            # Получение информации о символе
            self._get_symbol_info()

            # Загрузка начальных данных
            klines = self._retry_api_call(
                self.client.futures_klines,
                symbol=self.symbol,
                interval=self.interval,
                limit=self.data_window_size
            )
        
            for k in klines:
                self.data.loc[len(self.data)] = {
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                }
            # Обучение на исторических данных:
            if len(self.data) > 100 and (not hasattr(self, 'transformer_model') or self.transformer_model is None):
                logger.info("Обнаружены данные, но нет модели. Запускаю обучение...")
                if not self.train_model(epochs=10):  # Ускоренное обучение
                    logger.error("Не удалось обучить модель")
            # Работа с моделью
            if not hasattr(self, 'transformer_model') or self.transformer_model is None:
                self.load_model()  # Попытка загрузить существующую модель

            if len(self.data) > 100 and (not hasattr(self, 'transformer_model') or self.transformer_model is None):
                logger.info("Начинаю обучение модели на исторических данных...")
                if self._train_transformer_model():
                    self.save_model()
                    logger.info("Модель успешно обучена и сохранена")
                else:
                    logger.warning("Не удалось обучить модель")

            # Финальные проверки перед запуском
            required_attrs = ['transformer_model', 'client', 'ws_manager']
            missing = [attr for attr in required_attrs if not hasattr(self, attr)]
            if missing:
                logger.error(f"Отсутствуют критичные атрибуты: {missing}")
                return False
            
            # Добавить проверку:
            logger.info(f"Первые 3 строки данных:\n{self.data.head(3).to_string()}")
            logger.info(f"Последние 3 строки данных:\n{self.data.tail(3).to_string()}")
            logger.info(f"Проверка NaN: {self.data.isnull().sum()}")
            
            # Запуск WebSocket
            if not self.start_websocket():
                logger.error("Не удалось запустить WebSocket")
                return False

            # Логирование успешного запуска
            logger.info(f"""Бот успешно инициализирован:
            ┌ Модель: {'загружена' if hasattr(self, 'transformer_model') and self.transformer_model is not None else 'не загружена'}
            ├ Свечей в памяти: {len(self.data)}
            ├ Пара: {self.symbol}
            └ Плечо: {self.leverage}x
            """)

            # После всех проверок
            if not os.path.exists('transformer_model.pth'):
                logger.warning("Принудительно сохраняю модель...")
                self.save_model()
    
            # Отправка уведомления о запуске
            start_msg = (
                f"🤖 <b>Агрессивный бот запущен</b>\n"
                f"┌ Пара: {self.symbol}\n"
                f"├ Таймфрейм: {self.interval}\n"
                f"├ Плечо: {self.leverage}x\n"
                f"├ Риск: {self.risk_percent*100}% на сделку\n"
                f"├ TP/SL: {self.take_profit*100:.1f}%/{self.stop_loss*100:.1f}%\n"
                f"├ Размер окна данных: {self.data_window_size} свечей\n"
                f"├ MinQty: {self.min_qty}\n"
                f"└ Стартовый баланс: {self.starting_balance:.2f} USDT"
            )
        
            self._send_telegram_alert(start_msg)
            logger.info(f"Агрессивный бот запущен для {self.symbol}")
            return True

        except Exception as e:
            logger.error(f"Критическая ошибка запуска: {str(e)}", exc_info=True)
            self._send_telegram_alert(f"❌ Критическая ошибка запуска: {str(e)}")
            return False

if __name__ == "__main__":
    console = Console()
    bot = None  # Инициализация переменной для корректной обработки ошибок

    try:
        with console.status("[bold green]Запуск бота...") as status:
            # 1. Инициализация бота
            try:
                bot = AggressiveFuturesBot()
            except Exception as init_error:
                console.print(f"⛔ Ошибка инициализации бота: {str(init_error)}", style="bold red")
                logging.critical(f"Ошибка инициализации бота: {str(init_error)}", exc_info=True)
                raise SystemExit(1)

            # 2. Запуск бота
            if not bot.start():
                console.print("⛔ Не удалось запустить бота", style="bold red")
                bot._send_telegram_alert("🆘 Критическая ошибка: бот не запущен")
                raise SystemExit(1)

            console.print("✅ Бот успешно запущен", style="bold green")
            
            # 3. Основной цикл работы
            try:
                last_health_check = time.time()
                while True:
                    time.sleep(1)
                    
                    # Периодическая проверка состояния (каждые 30 секунд)
                    current_time = time.time()
                    if current_time - last_health_check > 30:
                        if not bot._check_websocket():
                            console.print("⚠️ Проблема с WebSocket, перезапуск...", style="bold yellow")
                            try:
                                if not bot.restart():
                                    console.print("⛔ Не удалось перезапустить бота", style="bold red")
                                    break
                            except Exception as restart_error:
                                console.print(f"⛔ Ошибка перезапуска: {str(restart_error)}", style="bold red")
                                break
                        last_health_check = current_time
                        
            except KeyboardInterrupt:
                console.print("\n🛑 Получен сигнал прерывания...", style="bold yellow")
            except Exception as main_loop_error:
                console.print(f"💥 Ошибка в основном цикле: {str(main_loop_error)}", style="bold red")
                logging.critical(f"Ошибка в основном цикле: {str(main_loop_error)}", exc_info=True)
            finally:
                # 4. Корректное завершение работы
                console.print("⏳ Остановка бота...", style="bold yellow")
                try:
                    if bot:
                        bot.stop_websocket()
                        console.print("✅ Бот успешно остановлен", style="bold green")
                except Exception as stop_error:
                    console.print(f"⛔ Ошибка при остановке: {str(stop_error)}", style="bold red")
                    logging.error(f"Ошибка при остановке бота: {str(stop_error)}", exc_info=True)
            #5 Обучение на исторических данных
            if bot.start():
                    if bot.train_model(epochs=15):
                        bot._send_telegram_alert("✅ Модель успешно обучена")
                    else:
                        bot._send_telegram_alert("❌ Ошибка обучения модели")
            #6 Для принудительного обучения на исторических данных
            if bot.train_model(epochs=15):
                logger.info("Модель успешно обучена")
            else:
                logger.error("Ошибка обучения модели")

            # Проверка подготовки данных
            x, y, _ = bot._prepare_transformer_data()
            if x is not None:
                print(f"Данные подготовлены успешно. X shape: {x.shape}")
                bot.train_model()

            if __name__ == "__main__":
                bot = AggressiveFuturesBot()
                if bot.load_historical_data(days=14):
                    print(f"Загружено {len(bot.data)} свечей")
                    print(bot.data[['timestamp', 'close']].tail())
    except SystemExit:
        pass  # Уже обработано выше
    except Exception as outer_error:
        console.print(f"💥 Необработанная ошибка: {str(outer_error)}", style="bold red")
        logging.critical(f"Необработанная ошибка: {str(outer_error)}", exc_info=True)
    finally:
        # 5. Финализация
        if bot:
            try:
                bot._send_telegram_alert("🔴 Бот остановлен")
            except Exception as alert_error:
                console.print(f"⚠️ Не удалось отправить уведомление: {str(alert_error)}", style="bold yellow")    
