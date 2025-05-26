# dashboard/tasks.py
import logging
from django.contrib.auth import get_user_model
from channels.layers import get_channel_layer
from utils.websockets import send_update_to_user_sync, broadcast_public_message_sync # 导入同步发送函数
from chaoyue_dreams.celery import app as celery_app
import time
import random
from users.models import FavoriteStock

logger = logging.getLogger(__name__)
User = get_user_model()

@celery_app.task(bind=True, name='dashboard.tasks.send_update_to_user_task_celery', queue='dashboard')
def send_update_to_user_task_celery(user_id: int, sub_type: str, payload: dict):
    """
    Celery异步任务，调用WebSocketSender类的send_update_to_user_task方法推送消息。
    """
    send_update_to_user_sync(user_id, sub_type, payload)

@celery_app.task(bind=True, name='dashboard.tasks.push_realtime_updates_for_stocks')
def push_realtime_updates_for_stocks(updated_stock_codes: list):
    """
    Celery任务：根据更新的股票代码列表，找到相关用户，获取其自选股最新数据并通过WebSocket推送。
    """
    if not updated_stock_codes:
        logger.info("没有更新的股票代码，跳过推送任务。")
        return

    channel_layer = get_channel_layer()
    if channel_layer is None:
        logger.error("无法获取 Channel Layer，WebSocket推送失败。")
        return

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
