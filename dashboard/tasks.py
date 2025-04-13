# dashboard/tasks.py
from celery import shared_task
from .consumers import send_update_to_user_sync, broadcast_public_message_sync # 导入同步发送函数
import time
import random
from users.models import CustomUser, FavoriteStock

@shared_task
def generate_random_strategy_message():
    """示例任务：生成随机策略消息并广播"""
    messages = [
        "策略C - 601318 中国平安 - 撤销买入挂单",
        "策略A - 000725 京东方A - 持仓中，等待信号",
        "策略D - 300750 宁德时代 - 发现潜在买点，监控中...",
        "市场情绪指标达到高位，注意风险",
    ]
    message_text = random.choice(messages)
    payload = {
        'timestamp': time.strftime("%H:%M:%S"),
        'text': message_text,
        # 可以添加更多结构化信息
    }
    print(f"Broadcasting strategy message: {message_text}")
    broadcast_public_message_sync(sub_type='strategy_message', payload=payload)

@shared_task
def update_favorite_stock_data():
    """示例任务：模拟更新所有用户的自选股数据"""
    users = CustomUser.objects.filter(is_active=True) # 获取所有活跃用户
    for user in users:
        favorites = FavoriteStock.objects.filter(user=user).select_related('stock')
        if not favorites.exists():
            continue

        # 模拟为每个用户的自选股生成更新数据
        for fav in favorites:
            # --- 在真实场景中，这里会调用行情接口获取实时数据 ---
            new_price = round(random.uniform(10, 100), 2)
            change = round(random.uniform(-2, 2), 2)
            volume = random.randint(1000, 500000)
            signals = [
                {'type': 'buy', 'text': '策略A: 买入'},
                {'type': 'sell', 'text': '策略B: 卖出'},
                {'type': 'hold', 'text': '持有中'},
                {'type': 'alert', 'text': '价格预警'},
            ]
            current_signal = random.choice(signals)
            # --- 模拟结束 ---

            update_payload = {
                'code': fav.stock.code,
                'latest_price': new_price,
                'change_percent': change,
                'volume': volume,
                'signal': current_signal,
            }
            print(f"Sending update for {fav.stock.code} to user {user.id}")
            # 发送给特定用户
            send_update_to_user_sync(user_id=user.id, sub_type='stock_update', payload=update_payload)

            # 模拟生成自选股相关的消息
            if random.random() < 0.1: # 10% 的概率生成消息
                 fav_message_payload = {
                     'timestamp': time.strftime("%H:%M:%S"),
                     'code': fav.stock.code,
                     'name': fav.stock.name,
                     'signal_type': current_signal['type'],
                     'text': f"{current_signal['text']} @ {new_price}"
                 }
                 print(f"Sending favorite message for {fav.stock.code} to user {user.id}")
                 send_update_to_user_sync(user_id=user.id, sub_type='favorite_message', payload=fav_message_payload)


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
