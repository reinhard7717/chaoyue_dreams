# dashboard/routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # 定义 WebSocket 连接的路径，映射到 Consumer
    re_path(r'ws/dashboard/$', consumers.DashboardConsumer.as_asgi()),
]
