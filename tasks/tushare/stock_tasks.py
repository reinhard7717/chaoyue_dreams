# stock_data_app/tasks.py
import asyncio
from asgiref.sync import async_to_sync
from utils.task_helpers import with_cache_manager
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from utils.cache_manager import CacheManager
from utils.websockets import send_update_to_user_sync # 导入推送函数
import logging

# 自选股队列
FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_SaveData_TimeTrade'
STOCKS_SAVE_API_DATA_QUEUE = 'SaveData_TimeTrade'
logger = logging.getLogger('tasks')

@celery_app.task(bind=True, name='tasks.tushare.stock_tasks.save_stock_list_data')
@with_cache_manager
def save_stock_list_data(self, cache_manager: CacheManager):
    """
    保存股票列表数据
    """
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    async def main():
        print("开始保存股票列表数据...")
        result = await stock_basic_dao.save_stocks()
        print(f"保存股票列表数据成功: {result}")
        result = await stock_basic_dao.save_company_info()
        print(f"保存公司信息数据成功: {result}")
        result = await stock_basic_dao.save_hs_const()
        print(f"保存沪深港通数据成功: {result}")
    async_to_sync(main)()


@celery_app.task(bind=True, name='tasks.tushare.stock_tasks.fetch_data_for_new_favorite')
@with_cache_manager
def fetch_data_for_new_favorite(self, user_id: int, stock_code: int, favorite_id: int, cache_manager: CacheManager):
    """
    为新添加的自选股获取实时数据和信号，并推送给用户。
    """
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    logger.info(f"开始为用户 {user_id} 的新自选股 {stock_code} (Favorite ID: {favorite_id}) 获取数据...")
    async def main():
        realtime_dao = StockRealtimeDAO(cache_manager)
        strategies_dao = StrategiesDAO(cache_manager)
        # 1. 获取股票基本信息 (code, name)
        stock_info = await stock_basic_dao.get_stock_by_code(stock_code)
        if not stock_info:
            logger.error(f"无法找到 stock_id={stock_code} 的股票信息")
            return
        # 2. 获取最新实时数据 (优先从缓存)
        latest_data = await realtime_dao.get_latest_tick_data(stock_code)
        current_price = latest_data.get('current_price') if latest_data else None
        high_price = latest_data.get('high_price') if latest_data else None
        low_price = latest_data.get('low_price') if latest_data else None
        open_price = latest_data.get('open_price') if latest_data else None
        prev_close_price = latest_data.get('prev_close_price') if latest_data else None
        trade_time = latest_data.get('trade_time') if latest_data else None
        turnover_value = latest_data.get('turnover_value') if latest_data else None
        volume = latest_data.get('volume') if latest_data else None
        change_percent = latest_data.get('change_percent') if latest_data else None
        # 3. 获取最新策略信号 (优先从缓存)
        latest_strategy_result = await strategies_dao.get_latest_strategy_result(stock_code)
        score = getattr(latest_strategy_result, 'score', None)
        if score is None:
            signal_type = 'hold'
            signal_text = 'N/A'
        else:
            if score >= 75:
                signal_type = 'buy'
            elif score <= 25:
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            signal_text = str(score)
        signal = {
            'type': signal_type,
            'text': signal_text
        }
        # 4. 组装 Payload
        payload_data = {
            'id': favorite_id,
            'code': stock_info.stock_code,
            'name': stock_info.stock_name,
            'current_price': current_price,
            'high_price': high_price,
            'low_price': low_price,
            'open_price': open_price,
            'prev_close_price': prev_close_price,
            'trade_time': trade_time,
            'turnover_value': turnover_value,
            'volume': volume,
            'change_percent': change_percent,
            'signal': signal,
        }
        print(f"payload_data: {payload_data}")
        # 5. 推送 WebSocket 消息
        send_update_to_user_sync(
            user_id=user_id,
            sub_type='favorite_added_with_data',
            payload=payload_data
        )
        logger.info(f"成功推送新自选股 {stock_info.stock_code} 数据给用户 {user_id}")
    async_to_sync(main)()
