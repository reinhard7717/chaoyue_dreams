from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _


class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'
    verbose_name = _('用户管理')
    def ready(self):
        """
        导入信号处理器
        """
        import users.signals  # 导入信号处理模块
