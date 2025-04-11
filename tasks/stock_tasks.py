# stock_data_app/tasks.py
import asyncio
import logging
from celery import shared_task
from django.db import models # 导入 models 以便 sync_to_async 可以找到它
from asgiref.sync import sync_to_async
import math

from core.constants import TIME_TEADE_TIME_LEVELS_LITE

# 获取 logger 实例
logger = logging.getLogger('celery') # 或者使用你项目配置的 logger

