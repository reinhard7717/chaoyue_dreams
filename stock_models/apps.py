from django.apps import AppConfig


class StockModelsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stock_models'
    verbose_name = '股票模型'

    def ready(self):
        from . import stock_basic, stock_realtime, index, fund_flow, stock_indicators, stock_analytics
        from .datacenter import capital_flow, financial, institution, lhb, market_data, north_south, statistics
        # from .indicator import adl, atr, boll, cci, cmf, dmi, ichimoku, kc, kdj, ma, macd, mfi, mom, obv, pivot_points, roc, rsi, sar, sma, stochastic_oscillator, vroc, vwap, wr