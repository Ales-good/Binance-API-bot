# utils/websocket_client.py
from binance import ThreadedWebsocketManager
import requests
from typing import Optional, Dict, Callable

def send_telegram_message(message: str, bot_token: str, chat_id: str) -> None:
    """
    Отправляет сообщение через Telegram.
    :param message: Текст сообщения
    :param bot_token: Токен Telegram-бота
    :param chat_id: ID чата
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    requests.post(url, json=payload)

class WebSocketManager:
    def __init__(self):
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
    
    def start_websocket(self, 
                      symbol: str, 
                      interval: str, 
                      callback: Callable, 
                      telegram_config: Optional[Dict] = None) -> None:
        """
        Запускает WebSocket для получения данных о свечах.
        :param symbol: Символ (например, "BTCUSDT")
        :param interval: Интервал свечей (например, "1m")
        :param callback: Функция, которая будет вызвана при получении данных
        :param telegram_config: Словарь с настройками Telegram (bot_token, chat_id)
        """
        def handle_message(msg):
            if msg['e'] == 'kline':  # Если это сообщение о свече
                kline = msg['k']
                if kline['x']:  # Если свеча закрылась
                    close_price = float(kline['c'])
                    volume = float(kline['v'])
                    
                    # Фильтрация по объёму торгов
                    if volume > 100:  # Пример: только крупные объёмы
                        callback(msg)
                        
                        # Отправка уведомления в Telegram
                        if telegram_config:
                            bot_token = telegram_config.get("bot_token")
                            chat_id = telegram_config.get("chat_id")
                            send_telegram_message(
                                f"Новое закрытие {symbol}: {close_price} (объём: {volume})",
                                bot_token,
                                chat_id
                            )

        self.twm.start_kline_socket(
            symbol=symbol,
            interval=interval,
            callback=handle_message
        )
    
    def stop(self) -> None:
        """Останавливает WebSocket соединение"""
        self.twm.stop()

# Пример использования:
if __name__ == "__main__":
    def print_message(msg):
        print("Получены данные:", msg)
    
    telegram_config = {
        "bot_token": "YOUR_BOT_TOKEN",
        "chat_id": "YOUR_CHAT_ID"
    }
    
    ws_manager = WebSocketManager()
    ws_manager.start_websocket(
        symbol="BTCUSDT",
        interval="1m",
        callback=print_message,
        telegram_config=telegram_config
    )
    
    # Для остановки (в реальном коде по какому-то условию)
    # ws_manager.stop()