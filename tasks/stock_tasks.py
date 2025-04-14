# stock_data_app/tasks.py
from celery import shared_task
from django.core.cache import cache # 或者你的 CacheManager
from utils.websockets import send_update_to_user_sync # 导入推送函数
import logging

# 获取 logger 实例
logger = logging.getLogger('tasks') # 或者使用你项目配置的 logger

def fetch_data_for_new_favorite(self, user_id: int, stock_code: int, favorite_id: int):
    """
    为新添加的自选股获取实时数据和信号，并推送给用户。
    """
    logger.info(f"开始为用户 {user_id} 的新自选股 {stock_code} (Favorite ID: {favorite_id}) 获取数据...")
    from dao_manager.daos.stock_basic_dao import StockBasicDAO
    from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO # 假设有获取最新数据的方法
    from dao_manager.daos.strategies_dao import StrategiesDAO
    try:
        # 实例化 DAOs (考虑使用依赖注入或更好的方式管理实例)
        stock_basic_dao = StockBasicDAO()
        realtime_dao = StockRealtimeDAO()
        strategies_dao = StrategiesDAO()

        # 1. 获取股票基本信息 (code, name)
        stock_info = stock_basic_dao.get_stock_by_code(stock_code) # 假设 UserDAO 有此方法
        if not stock_info:
            logger.error(f"无法找到 stock_id={stock_code} 的股票信息")
            return

        # 2. 获取最新实时数据 (优先从缓存)
        # 注意：这些 get 方法需要你自己实现，以下是示例逻辑
        latest_data = realtime_dao.get_latest_realtime_data(stock_code) # 返回包含 price, volume, change_percent 等的字典或对象
        latest_price = latest_data.get('price') if latest_data else None
        volume = latest_data.get('volume') if latest_data else None
        change_percent = latest_data.get('change_percent') if latest_data else None # 假设 DAO 能提供涨跌幅

        # 3. 获取最新策略信号 (优先从缓存)
        signal_data = strategies_dao.get_latest_strategies(stock_code) # 返回包含 type 和 text 的字典或对象
        signal = {
            'type': signal_data.get('signal_display', 'hold'), 
            'text': signal_data.get('text', 'N/A')
            } if signal_data else {'type': 'hold', 'text': 'N/A'}

        # 4. 组装 Payload
        payload_data = {
            'id': favorite_id, # 使用 FavoriteStock 的 ID
            'code': stock_info.stock_code,
            'name': stock_info.stock_name,
            'latest_price': latest_price,
            'change_percent': change_percent,
            'volume': volume,
            'signal': signal,
        }

        # 5. 推送 WebSocket 消息
        send_update_to_user_sync(
            user_id=user_id,
            sub_type='favorite_added_with_data', # 使用新的子类型
            payload=payload_data
        )
        logger.info(f"成功推送新自选股 {stock_info.stock_code} 数据给用户 {user_id}")

    except Exception as e:
        logger.error(f"为新自选股 {stock_code} 获取数据并推送时出错: {e}", exc_info=True)

    finally:
        # 如果 DAO 需要关闭连接，在这里处理 (取决于你的 DAO 实现)
        realtime_dao.close()