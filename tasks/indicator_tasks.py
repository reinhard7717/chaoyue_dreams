import asyncio
import logging
from celery import group
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from services.indicator_services import IndicatorService
from core.constants import TIME_TEADE_TIME_LEVELS, TimeLevel
from chaoyue_dreams.celery import app as celery_app
from tasks.stock_indicator_tasks import calculate_stock_indicators_for_single_stock

logger = logging.getLogger('tasks')
