# tasks\tushare\industry_tasks.py

import os
import logging
# 假设 StockBasicInfoDao 存在且可用
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
import pandas as pd
import asyncio
# 假设 celery 实例存在且可用
from chaoyue_dreams.celery import app as celery_app


logger = logging.getLogger("tasks")

# 任务：准备 Transformer 训练数据并保存
@celery_app.task(bind=True, name='tasks.tushare.industry_tasks.save_ths_index_list_task')
def save_ths_index_list_task(self):
    industry_dao = IndustryDao()
    logger.info(f"开始获取同花顺板块指数...")
    result = asyncio.run(industry_dao.save_ths_index_list())
