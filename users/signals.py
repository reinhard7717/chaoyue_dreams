from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth import get_user_model

User = get_user_model()

@receiver(post_save, sender=User)
def user_created_handler(sender, instance, created, **kwargs):
    """
    用户创建后的信号处理
    """
    if created:
        # 用户创建后的处理逻辑
        pass 