from django.apps import AppConfig


class StockModelsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stock_models'
    verbose_name = '股票模型'
    # def ready(self):