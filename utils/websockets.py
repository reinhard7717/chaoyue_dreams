# utils/websockets.py (或者 services/websocket_service.py 等)

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import logging

logger = logging.getLogger(__name__)

def send_update_to_user_sync(user_id: int, sub_type: str, payload: dict):
    """
    同步地向指定用户的 WebSocket 组发送消息。
    Args:
        user_id: 目标用户的 ID.
        sub_type: 消息的子类型 (例如 'stock_update', 'favorite_message', 'favorites_update').
        payload: 要发送的数据字典.
    """
    channel_layer = get_channel_layer()
    group_name = f'user_{user_id}'
    message_data = {
        'type': 'user.message', # 对应 Consumer 中的 user_message 方法
        'data': {
            'sub_type': sub_type,
            'payload': payload,
        }
    }
    try:
        async_to_sync(channel_layer.group_send)(group_name, message_data)
        logger.debug(f"成功发送消息到组 {group_name} (sub_type: {sub_type})")
    except Exception as e:
        # 在实际发送处也记录错误可能更有用
        logger.error(f"发送消息到组 {group_name} 失败: {e}", exc_info=False)


def broadcast_public_message_sync(sub_type: str, payload: dict):
    """
    同步地向公共 WebSocket 组广播消息。
    Args:
        sub_type: 消息的子类型 (例如 'strategy_message', 'market_alert').
        payload: 要发送的数据字典.
    """
    channel_layer = get_channel_layer()
    group_name = 'dashboard_public' # 公共组名
    message_data = {
        'type': 'public.message', # 对应 Consumer 中的 public_message 方法
        'data': {
            'sub_type': sub_type,
            'payload': payload,
        }
    }
    try:
        async_to_sync(channel_layer.group_send)(group_name, message_data)
        logger.debug(f"成功广播消息到组 {group_name} (sub_type: {sub_type})")
    except Exception as e:
        logger.error(f"广播消息到组 {group_name} 失败: {e}", exc_info=False)

# 如果需要在异步代码中使用，可以定义异步版本
# async def send_update_to_user_async(user_id: int, sub_type: str, payload: dict):
#     channel_layer = get_channel_layer()
#     group_name = f'user_{user_id}'
#     message_data = { ... }
#     try:
#         await channel_layer.group_send(group_name, message_data)
#         logger.debug(...)
#     except Exception as e:
#         logger.error(...)
