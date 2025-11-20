from django.db import models
from django.conf import settings
# from django.contrib.auth.models import User
from django.utils.translation import gettext_lazy as _
from django.db.models.signals import post_save
from django.dispatch import receiver

from stock_models.stock_basic import StockInfo

# 先使用Django自带的用户模型
# class User(AbstractUser):
#     """
#     用户模型，扩展Django自带的用户模型
#     """
#     # 用户手机号码
#     phone = models.CharField(_('手机号码'), max_length=11, blank=True, null=True)
#     # 用户头像
#     avatar = models.ImageField(_('用户头像'), upload_to='avatars/', blank=True, null=True)
#     # 用户简介
#     bio = models.TextField(_('个人简介'), max_length=500, blank=True)
#     # 最后登录IP
#     last_login_ip = models.GenericIPAddressField(_('最后登录IP'), blank=True, null=True)
#     # 是否接收邮件通知
#     email_notification = models.BooleanField(_('邮件通知'), default=True)
#     # 创建时间和更新时间
#     created_at = models.DateTimeField(_('创建时间'), auto_now_add=True)
#     updated_at = models.DateTimeField(_('更新时间'), auto_now=True)

#     class Meta:
#         verbose_name = _('用户')
#         verbose_name_plural = _('用户')
#         ordering = ['-date_joined']

#     def __str__(self):
#         return self.username
    
#     def get_full_name(self):
#         """
#         返回用户全名
#         """
#         return f"{self.first_name} {self.last_name}" if self.first_name and self.last_name else self.username

class UserProfile(models.Model):
    """
    用户资料模型，扩展Django自带的用户模型
    """
    # 关联Django自带的用户模型
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, # <--- 使用字符串引用
        on_delete=models.CASCADE,
        related_name='profile',
        verbose_name='用户',
        primary_key=True, # OneToOneField 通常设为主键以保证唯一性
    )
    # 用户手机号码
    phone = models.CharField(_('手机号码'), max_length=11, blank=True, null=True)
    # 用户头像
    avatar = models.ImageField(_('用户头像'), upload_to='avatars/', blank=True, null=True)
    # 用户简介
    bio = models.TextField(_('个人简介'), max_length=500, blank=True)
    # 最后登录IP
    last_login_ip = models.GenericIPAddressField(_('最后登录IP'), blank=True, null=True)
    # 是否接收邮件通知
    email_notification = models.BooleanField(_('邮件通知'), default=True)
    # 创建时间和更新时间
    created_at = models.DateTimeField(_('创建时间'), auto_now_add=True)
    updated_at = models.DateTimeField(_('更新时间'), auto_now=True)
    class Meta:
        verbose_name = _('用户资料')
        verbose_name_plural = _('用户资料')
    def __str__(self):
        # 确保 self.user 存在再访问 username，虽然 OneToOne 理论上 user 总存在
        return self.user.username if hasattr(self, 'user') and self.user else f"Profile for User ID {self.pk}"

class FavoriteStock(models.Model):
    """
    自选股模型，用于存储用户的自选股
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, # <--- 使用字符串引用
        on_delete=models.CASCADE,
        related_name='favorite_stocks',
        verbose_name='用户'
    )
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, blank=True, null=True, related_name="favorite_stocks", verbose_name=_("股票"))
    # 添加时间
    added_at = models.DateTimeField(_('添加时间'), auto_now_add=True)
    # 备注
    note = models.CharField(_('备注'), max_length=200, blank=True)
    # 是否置顶
    is_pinned = models.BooleanField(_('是否置顶'), default=False)
    # 标签，用于分组
    tags = models.CharField(_('标签'), max_length=200, blank=True)
    class Meta:
        verbose_name = _('自选股')
        verbose_name_plural = _('自选股')
        ordering = ['-is_pinned', '-added_at']
        # 确保一个用户不会添加重复的股票
        unique_together = ['user', 'stock']
    def __str__(self):
        return f"{self.user.username} - {self.stock.stock_name}({self.stock.stock_code})"

# 监听信号，当用户创建后自动创建用户资料
@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_user_profile(sender, instance, created, **kwargs):
    """
    当用户创建后的操作
    """
    if created:
        UserProfile.objects.create(user=instance)

