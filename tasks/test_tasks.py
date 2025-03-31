"""
测试任务模块
提供简单的测试任务，用于验证Celery的正常工作
"""
import logging
from celery import shared_task
import time

logger = logging.getLogger(__name__)

@shared_task
def add(x, y):
    """
    简单的加法测试任务
    """
    logger.info(f"正在执行加法任务: {x} + {y}")
    # 添加一个短暂的延迟，模拟任务执行
    time.sleep(2)
    result = x + y
    logger.info(f"加法任务完成: {x} + {y} = {result}")
    return result

@shared_task
def hello_world():
    """
    简单的问候测试任务
    """
    logger.info("正在执行问候任务")
    time.sleep(1)
    message = "你好，世界！这是一个测试任务。"
    logger.info(f"问候任务完成: {message}")
    return message

@shared_task
def long_task(seconds=10):
    """
    长时间运行的测试任务
    """
    logger.info(f"开始执行长时间任务，将运行{seconds}秒")
    for i in range(seconds):
        time.sleep(1)
        logger.info(f"长时间任务进度: {i+1}/{seconds}")
    logger.info("长时间任务完成")
    return f"长时间任务完成，运行了{seconds}秒" 