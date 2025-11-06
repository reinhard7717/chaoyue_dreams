# 文件: dashboard/signals.py

from django.db.models.signals import post_delete
from django.dispatch import receiver
from users.models import FavoriteStock
from stock_models.stock_analytics import PositionTracker
import logging

logger = logging.getLogger(__name__)

@receiver(post_delete, sender=FavoriteStock)
def delete_position_tracker_on_favorite_delete(sender, instance, **kwargs):
    """
    【V2.0 - 适配 PositionTracker 模型】
    当一个 FavoriteStock 记录被删除后，自动删除所有相关的 PositionTracker 记录。
    由于模型设置了级联删除 (on_delete=models.CASCADE)，删除 PositionTracker
    会自动清理其下所有的 Transaction 和 DailyPositionSnapshot 记录。
    """
    # 从被删除的 FavoriteStock 实例中获取用户和股票信息
    user = instance.user
    stock = instance.stock
    # 查找与该用户和股票对应的 PositionTracker
    # 注意：根据模型定义，每个用户/股票组合只有一个 PositionTracker，但使用 .filter() 更安全
    trackers_to_delete = PositionTracker.objects.filter(user=user, stock=stock)
    # 执行删除操作，并获取删除的记录数
    # 这会触发数据库的级联删除，一并删除所有关联的交易和快照
    deleted_count, _ = trackers_to_delete.delete()
    # 如果确实删除了记录，则打印日志
    if deleted_count > 0:
        logger.info(
            f"[Signal] 因 FavoriteStock 删除，联动删除了 {deleted_count} 条 "
            f"属于用户 {user.username} 的关于股票 {stock.stock_code} 的 PositionTracker "
            f"及其所有关联的交易和快照记录。"
        )

