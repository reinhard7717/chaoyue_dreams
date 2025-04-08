# 示例：在一个异步视图或任务中
from services.indicator_services import IndicatorService
from core.constants import TimeLevel

async def update_stock_indicators(stock_code: str, time_level_str: str):
    service = IndicatorService()
    # 可以将字符串转换为 TimeLevel 枚举，如果需要的话
    # time_level_enum = TimeLevel(time_level_str)
    await service.calculate_and_save_all_indicators(stock_code, time_level_str)

# 调用示例
# await update_stock_indicators('000001', '1d')
# await update_stock_indicators('600519', TimeLevel.MIN_60)
