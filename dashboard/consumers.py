# dashboard/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async # 异步访问数据库
from users.models import FavoriteStock

class DashboardConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]

        if not self.user.is_authenticated:
            await self.close() # 拒绝未认证用户
            return

        # 每个用户加入自己的私有组，用于接收个人消息（如自选股更新）
        self.user_group_name = f'user_{self.user.id}'
        await self.channel_layer.group_add(
            self.user_group_name,
            self.channel_name
        )

        # （可选）所有认证用户加入一个公共组，用于接收广播消息（如通用策略信号）
        self.public_group_name = 'dashboard_public'
        await self.channel_layer.group_add(
            self.public_group_name,
            self.channel_name
        )

        await self.accept()
        print(f"WebSocket connected for user {self.user.username}")

        # 连接成功后，可以立即发送一些初始状态，例如完整的自选股列表
        # await self.send_initial_favorites()

    async def disconnect(self, close_code):
        if hasattr(self, 'user_group_name'):
            await self.channel_layer.group_discard(
                self.user_group_name,
                self.channel_name
            )
        if hasattr(self, 'public_group_name'):
             await self.channel_layer.group_discard(
                self.public_group_name,
                self.channel_name
            )
        print(f"WebSocket disconnected for user {self.user.username}")

    # 从 WebSocket 接收消息 (前端发送过来的，这个场景可能用得少)
    async def receive(self, text_data):
        # text_data_json = json.loads(text_data)
        # message = text_data_json['message']
        # print(f"Received message from {self.user.username}: {message}")
        # 可以根据前端发来的指令做些事情，但不推荐用 WebSocket 做 API 调用
        pass

    # --- 处理从 Channel Layer 发送到 Group 的消息 ---
    # 处理发送到 user_{id} 组的消息
    async def user_message(self, event):
        message_type = event.get('type') # 'user.message' -> 'message'
        data = event.get('data', {})
        message_sub_type = data.get('sub_type') # 例如 'favorite_update', 'private_signal'

        # 将消息发送给 WebSocket 客户端
        await self.send(text_data=json.dumps({
            'type': message_sub_type, # 使用子类型作为前端判断依据
            'payload': data.get('payload', {})
        }))

    # 处理发送到 dashboard_public 组的消息
    async def public_message(self, event):
        message_type = event.get('type') # 'public.message' -> 'message'
        data = event.get('data', {})
        message_sub_type = data.get('sub_type') # 例如 'general_signal', 'market_alert'

        await self.send(text_data=json.dumps({
            'type': message_sub_type,
            'payload': data.get('payload', {})
        }))

    # --- 辅助方法 (示例) ---
    # @database_sync_to_async
    # def get_initial_favorites_data(self):
    #     favorites = FavoriteStock.objects.filter(user=self.user).select_related('stock')
    #     # 这里需要序列化数据，但不能直接用 DRF 序列化器，需要手动构建或写简单序列化函数
    #     data = [
    #         {
    #             'code': fav.stock.code,
    #             'name': fav.stock.name,
    #             # ... 其他需要初始加载的数据 ...
    #             'latest_price': None, # 初始可以为空，等后续推送
    #             'change_percent': None,
    #             'volume': None,
    #             'signal': None,
    #         } for fav in favorites
    #     ]
    #     return data

    # async def send_initial_favorites(self):
    #     favorites_data = await self.get_initial_favorites_data()
    #     await self.send(text_data=json.dumps({
    #         'type': 'initial_favorites',
    #         'payload': favorites_data
    #     }))

# --- 如何从其他地方 (如 Celery 任务或 API 视图) 发送消息 ---
# from channels.layers import get_channel_layer
# from asgiref.sync import async_to_sync

# async def send_update_to_user(user_id, sub_type, payload):
#     channel_layer = get_channel_layer()
#     await channel_layer.group_send(
#         f'user_{user_id}',
#         {
#             'type': 'user.message', # 对应 consumer 中的 user_message 方法
#             'data': {
#                 'sub_type': sub_type,
#                 'payload': payload,
#             }
#         }
#     )

# def send_update_to_user_sync(user_id, sub_type, payload):
#     """同步版本，用于普通 Django 视图或 Celery 任务"""
#     channel_layer = get_channel_layer()
#     async_to_sync(channel_layer.group_send)(
#         f'user_{user_id}',
#         {
#             'type': 'user.message',
#             'data': {
#                 'sub_type': sub_type,
#                 'payload': payload,
#             }
#         }
#     )

# async def broadcast_public_message(sub_type, payload):
#      channel_layer = get_channel_layer()
#      await channel_layer.group_send(
#          'dashboard_public',
#          {
#              'type': 'public.message', # 对应 consumer 中的 public_message 方法
#              'data': {
#                  'sub_type': sub_type,
#                  'payload': payload,
#              }
#          }
#      )
# def broadcast_public_message_sync(sub_type, payload):
#      channel_layer = get_channel_layer()
#      async_to_sync(channel_layer.group_send)(
#          'dashboard_public',
#          {
#              'type': 'public.message',
#              'data': {
#                  'sub_type': sub_type,
#                  'payload': payload,
#              }
#          }
#      )

