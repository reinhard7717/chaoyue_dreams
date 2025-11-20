# chaoyue_dreams/asgi.py
import os
import django # 导入 django
from channels.auth import AuthMiddlewareStack # 用于 WebSocket 认证
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator # 限制来源
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chaoyue_dreams.settings')

# --- 在这里显式调用 django.setup() ---
django.setup()
# --- 调用结束 ---

# 导入 dashboard 的路由配置 
import dashboard.routing

application = ProtocolTypeRouter({
    # Django's ASGI application to handle traditional HTTP requests
    "http": get_asgi_application(),
    # WebSocket chat handler
    "websocket": AllowedHostsOriginValidator( # 生产环境建议使用
        AuthMiddlewareStack( # 处理 WebSocket 连接的用户认证
            URLRouter(
                dashboard.routing.websocket_urlpatterns # 指向 dashboard app 的 WebSocket 路由
            )
        )
    ),
})
