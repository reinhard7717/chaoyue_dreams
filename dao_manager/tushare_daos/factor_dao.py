# dao_manager\tushare_daos\factor_dao.py

import pandas as pd
import datetime
from asgiref.sync import sync_to_async
from typing import List, Dict, Optional
from django.db.models import QuerySet
from utils.model_helpers import (
    get_chip_factor_model_by_code, get_fundflow_factor_model_by_code, get_chip_holding_matrix_model_by_code
)

class FactorDao:
    def __init__(self):
        pass

    async def get_chip_factor_data(self, stock_code: str, trade_date: datetime.date, limit: int) -> pd.DataFrame:
        """
        从 ChipFactor 模型获取筹码因子数据。
        """
        model = get_chip_factor_model_by_code(stock_code)
        if not model:
            return pd.DataFrame()
        
        end_date = trade_date or datetime.date.today()
        qs = model.objects.filter(
            stock__stock_code=stock_code,
            trade_time__lte=end_date
        ).order_by('-trade_time')[:limit]
        
        df = await sync_to_async(lambda: pd.DataFrame.from_records(qs.values()))()
        
        if not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df = df.set_index('trade_time')
            df = df.sort_index()
            
        return df

    async def get_chip_holding_matrix_data(self, stock_code: str, trade_date: datetime.date, limit: int) -> pd.DataFrame:
        """
        从 ChipHoldingMatrix 模型获取筹码持有矩阵数据。
        """
        model = get_chip_holding_matrix_model_by_code(stock_code)
        if not model:
            return pd.DataFrame()
            
        end_date = trade_date or datetime.date.today()
        qs = model.objects.filter(
            stock__stock_code=stock_code,
            trade_time__lte=end_date
        ).order_by('-trade_time')[:limit]
        
        df = await sync_to_async(lambda: pd.DataFrame.from_records(qs.values()))()
        
        if not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df = df.set_index('trade_time')
            df = df.sort_index()
            
        return df

    async def get_fund_flow_factor_data(self, stock_code: str, trade_date: datetime.date, limit: int) -> pd.DataFrame:
        """
        从 FundFlowFactor 模型获取资金流向因子数据。
        """
        model = get_fundflow_factor_model_by_code(stock_code)
        if not model:
            return pd.DataFrame()
            
        end_date = trade_date or datetime.date.today()
        qs = model.objects.filter(
            stock__stock_code=stock_code,
            trade_time__lte=end_date
        ).order_by('-trade_time')[:limit]
        
        df = await sync_to_async(lambda: pd.DataFrame.from_records(qs.values()))()
        
        if not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df = df.set_index('trade_time')
            df = df.sort_index()
            
        return df












