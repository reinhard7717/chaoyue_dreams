# dashboard/signals.py
from django.db.models.signals import post_delete
from django.dispatch import receiver
from users.models import FavoriteStock
# from stock_models.stock_analytics import FavoriteStockTracker
import logging

logger = logging.getLogger(__name__)

@receiver(post_delete, sender=FavoriteStock)
def delete_tracker_on_favorite_delete(sender, instance, **kwargs):
    """
    当一个 FavoriteStock 记录被删除后，自动删除所有相关的 FavoriteStockTracker 记录。
    """
    user = instance.user
    stock = instance.stock
    
    trackers_to_delete = FavoriteStockTracker.objects.filter(user=user, stock=stock)
    deleted_count, _ = trackers_to_delete.delete()
    
    if deleted_count > 0:
        logger.info(
            f"[Signal] 因 FavoriteStock 删除，联动删除了 {deleted_count} 条 "
            f"属于用户 {user.username} 的关于股票 {stock.stock_code} 的追踪记录。"
        )
