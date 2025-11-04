# trading/orders.py
from binance.client import Client
from dotenv import load_dotenv
import os

load_dotenv()

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_SECRET_KEY'))

def create_order_with_sl_tp(symbol, side, quantity, stop_loss, take_profit):
    """
    Создаёт ордер с SL/TP.
    :param symbol: Символ (например, "BTCUSDT")
    :param side: Направление ("BUY" или "SELL")
    :param quantity: Количество актива
    :param stop_loss: Уровень стоп-лосса
    :param take_profit: Уровень тейк-профита
    """
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_STOP_LOSS_LIMIT,
            quantity=quantity,
            stopPrice=stop_loss,
            price=take_profit
        )
        print("Ордер с SL/TP создан:", order)
    except Exception as e:
        print("Ошибка при создании ордера:", e)