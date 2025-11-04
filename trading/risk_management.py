# trading/risk_management.py

def calculate_position_size(balance, risk_percentage, stop_loss_distance):
    """
    Рассчитывает размер позиции на основе риска.
    :param balance: Баланс в USDT
    :param risk_percentage: Процент риска (например, 0.01 для 1%)
    :param stop_loss_distance: Расстояние до стоп-лосса
    :return: Размер позиции
    """
    risk_amount = balance * risk_percentage
    position_size = risk_amount / stop_loss_distance
    return position_size