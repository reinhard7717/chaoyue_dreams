# stock_data_app/tasks.py
import asyncio
from celery import shared_task
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from utils.websockets import send_update_to_user_sync # 导入推送函数
from asgiref.sync import async_to_sync
import logging

# 自选股队列
FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_SaveData_RealTime'
STOCKS_SAVE_API_DATA_QUEUE = 'SaveData_RealTime'
logger = logging.getLogger('tasks')

@celery_app.task(bind=True, name='tasks.tushare.stock_tasks.save_stock_list_data')
def save_stock_list_data(self):
    """
    保存股票列表数据
    """
    stock_basic_dao = StockBasicInfoDao()
    result = asyncio.run(stock_basic_dao.save_stocks())
    logger.info(f"保存股票列表数据成功: {result}")
    result = asyncio.run(stock_basic_dao.save_company_info())
    logger.info(f"保存公司信息数据成功: {result}")
    result = asyncio.run(stock_basic_dao.save_hs_const())
    logger.info(f"保存沪深港通数据成功: {result}")

@celery_app.task(bind=True, name='tasks.tushare.stock_tasks.fetch_data_for_new_favorite')
def fetch_data_for_new_favorite(self, user_id: int, stock_code: int, favorite_id: int):
    """
    为新添加的自选股获取实时数据和信号，并推送给用户。
    """
    logger.info(f"开始为用户 {user_id} 的新自选股 {stock_code} (Favorite ID: {favorite_id}) 获取数据...")
    from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
    from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO # 假设有获取最新数据的方法
    from dao_manager.daos.strategies_dao import StrategiesDAO
    try:
        # 实例化 DAOs (考虑使用依赖注入或更好的方式管理实例)
        stock_basic_dao = StockBasicDAO()
        realtime_dao = StockRealtimeDAO()
        strategies_dao = StrategiesDAO()
        # 1. 获取股票基本信息 (code, name)
        stock_info = async_to_sync(stock_basic_dao.get_stock_by_code)(stock_code) # 假设 UserDAO 有此方法
        if not stock_info:
            logger.error(f"无法找到 stock_id={stock_code} 的股票信息")
            return
        # 2. 获取最新实时数据 (优先从缓存)
        latest_data = async_to_sync(realtime_dao.get_latest_realtime_data)(stock_code)
        current_price = latest_data.current_price if latest_data else None
        volume = latest_data.volume if latest_data else None
        price_change_percent = latest_data.price_change_percent if latest_data else None
        # 3. 获取最新策略信号 (优先从缓存)
        signal_data = async_to_sync(strategies_dao.get_latest_strategies)(stock_code) # 返回包含 type 和 text 的字典或对象
        signal = {
            'type': signal_data.get('signal_display', 'hold'), 
            'text': signal_data.get('text', 'N/A')
            } if signal_data else {'type': 'hold', 'text': 'N/A'}
        # 4. 组装 Payload
        payload_data = {
            'id': favorite_id, # 使用 FavoriteStock 的 ID
            'code': stock_info.stock_code,
            'name': stock_info.stock_name,
            'latest_price': current_price,
            'change_percent': price_change_percent,
            'volume': volume,
            'signal': signal,
        }
        print(f"payload_data: {payload_data}")
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