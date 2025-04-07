from django.apps import AppConfig


class StockModelsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stock_models'
    verbose_name = '股票模型'

    def ready(self):
        from . import stock_basic, stock_realtime, index, fund_flow, stock_indicators
        from .datacenter import capital_flow, financial, institution, lhb, market_data, north_south, statistics
        from .indicator import boll, kdj, ma, macd, obv, rsi, atr