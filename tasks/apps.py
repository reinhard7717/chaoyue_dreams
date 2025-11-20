# tasks/apps.py

import tushare as ts
from django.apps import AppConfig
from django.conf import settings # 导入Django的settings

class TasksConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tasks'
    def ready(self):
        """
        Django应用加载就绪后执行的钩子函数。
        这是执行一次性初始化任务（如设置API Token）的最佳位置。
        """
        print("Django App 'tasks' is ready. Initializing Tushare token...")
        try:
            if hasattr(settings, 'API_LICENCES_TUSHARE') and settings.API_LICENCES_TUSHARE:
                ts.set_token(settings.API_LICENCES_TUSHARE)
                print("Tushare token has been set successfully.")
            else:
                print("WARNING: TUSHARE_TOKEN not found in Django settings. Tushare API calls may fail.")
        except Exception as e:
            print(f"ERROR: Failed to set Tushare token during app startup: {e}")
        # 如果您有信号（signals）等，也应该在这里导入
        # import tasks.signals
