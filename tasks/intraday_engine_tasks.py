# 文件: tasks/intraday_engine_tasks.py
import logging
from chaoyue_dreams.celery import app as celery_app
from asgiref.sync import async_to_sync


logger = logging.getLogger("celery_tasks")

