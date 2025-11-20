# dashboard/consumers.py
import json
from django.core.serializers.json import DjangoJSONEncoder
from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async # 异步访问数据库
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from users.models import FavoriteStock
from utils.cache_manager import CacheManager

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
        """
        处理通用的、发送到用户私有组的消息。
        """
        data = event.get('data', {})
        message_sub_type = data.get('sub_type')
        payload = data.get('payload', {})
        # 将消息发送给 WebSocket 客户端
        await self.send(text_data=json.dumps({
            'type': message_sub_type, # 使用子类型作为前端判断依据
            'payload': payload
        }, cls=DjangoJSONEncoder))
    # 处理发送到 dashboard_public 组的消息
    async def public_message(self, event):
        message_type = event.get('type') # 'public.message' -> 'message'
        data = event.get('data', {})
        message_sub_type = data.get('sub_type') # 例如 'general_signal', 'market_alert'
        await self.send(text_data=json.dumps({
            'type': message_sub_type,
            'payload': data.get('payload', {})
        }))
    # 专门处理盘中引擎信号的方法
    async def intraday_signal_update(self, event):
        """
        处理由后端引擎通过Channel Layer发送来的盘中信号。
        这个方法名 (intraday_signal_update) 必须与 channel_layer.group_send 中
        指定的 'type' 完全匹配。
        """
        payload = event.get('payload', {})
        # 将信号封装成标准格式，发送给前端WebSocket客户端
        # 使用 DjangoJSONEncoder 可以安全地处理 datetime 等特殊类型
        await self.send(text_data=json.dumps({
            'type': 'intraday_signal_update', # 前端JS将根据这个type来识别消息
            'payload': payload
        }, cls=DjangoJSONEncoder))