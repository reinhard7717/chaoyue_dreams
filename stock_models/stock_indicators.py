from django.db import models
from django.utils.translation import gettext_lazy as _
from stock_models.stock_basic import StockInfo

# --- 指标模型 (使用斐波那契周期) ---
FIB_PERIODS = (5, 8, 13, 21, 34, 55, 89, 144, 233) # 定义斐波那契周期


