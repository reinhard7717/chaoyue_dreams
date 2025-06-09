# dashboard/tasks.py
import asyncio
from asgiref.sync import async_to_sync
import logging
from django.contrib.auth import get_user_model
from channels.layers import get_channel_layer
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from utils.websockets import send_update_to_user_sync, broadcast_public_message_sync # 导入同步发送函数
from chaoyue_dreams.celery import app as celery_app
import time
import random
from users.models import FavoriteStock
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO


logger = logging.getLogger(__name__)
User = get_user_model()

@celery_app.task(bind=True, name='dashboard.tasks.send_update_to_user_task_celery', queue='dashboard')
def send_update_to_user_task_celery(self, user_id: int, sub_type: str, payload: dict):
    """
    Celery异步任务，调用WebSocketSender类的send_update_to_user_task方法推送消息。
    """
    send_update_to_user_sync(user_id, sub_type, payload)

@celery_app.task(bind=True, name='dashboard.tasks.push_realtime_updates_for_stocks')
def push_realtime_updates_for_stocks(updated_stock_codes: list):
    """
    Celery任务：根据更新的股票代码列表，找到相关用户，获取其自选股最新数据并通过WebSocket推送。
    """
    from users.models import FavoriteStock
    from dashboard.tasks import send_update_to_user_task_celery
    stock_basic_dao = StockBasicInfoDao()
    stock_realtime_dao = StockRealtimeDAO()
    strategy_dao = StrategiesDAO()
    if not updated_stock_codes:
        logger.info("没有更新的股票代码，跳过推送任务。")
        return

    channel_layer = get_channel_layer()
    if channel_layer is None:
        logger.error("无法获取 Channel Layer，WebSocket推送失败。")
        return
    
    for stock_code in updated_stock_codes:
        stock_obj = async_to_sync(stock_basic_dao.get_stock_by_code)(stock_code)
        latest_tick = async_to_sync(stock_realtime_dao.get_latest_tick_data)(stock_code)
        latest_strategy_result = async_to_sync(strategy_dao.get_latest_strategy_result)(stock_code)
        # 获取所有关注该股票的用户
        user_ids = list(FavoriteStock.objects.filter(stock__stock_code=stock_code).values_list('user_id', flat=True))
        if not user_ids:
            # logger.info(f"股票{code}没有关注用户，跳过推送")
            continue
        # 获取最新tick数据（调用异步方法，转同步）
        
        if not latest_tick:
            logger.warning(f"未获取到股票{stock_obj}的最新tick数据，跳过推送")
            continue
        # --- 保证signal字段为对象 ---
        signal = latest_strategy_result.score
        if not isinstance(signal, dict):
            signal = {'type': 'hold', 'text': signal or 'N/A'}
        # --- 构造payload，字段名与前端updateStockRow完全一致 ---
        payload = {
            'code': stock_code,
            'current_price': latest_tick.get('current_price'),
            'high_price': latest_tick.get('high_price'),
            'low_price': latest_tick.get('low_price'),
            'open_price': latest_tick.get('open_price'),
            'prev_close_price': latest_tick.get('prev_close_price'),
            'trade_time': latest_tick.get('trade_time'),
            'turnover_value': latest_tick.get('turnover_value'),
            'volume': latest_tick.get('volume'),
            'change_percent': latest_tick.get("change_percent"),
            'signal': signal,
        }
        # 推送给所有关注该股票的用户
        for uid in user_ids:
            send_update_to_user_task_celery.apply_async(
                args=[uid, 'realtime_tick_update', payload],
                queue='dashboard'  # 指定队列为dashboard
            )
            # print(f"已推送{code}最新tick数据到用户{uid}")

# 你还需要配置 Celery Beat 来定时运行这些任务
# 例如，在 settings.py 中配置:
# CELERY_BEAT_SCHEDULE = {
#     'generate-strategy-message-every-10-seconds': {
#         'task': 'dashboard.tasks.generate_random_strategy_message',
#         'schedule': 10.0, # 每 10 秒
#     },
#      'update-favorites-every-5-seconds': {
#         'task': 'dashboard.tasks.update_favorite_stock_data',
#         'schedule': 5.0, # 每 5 秒
#     },
# }
