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

# 批量获取筹码因子的函数
async def get_chip_factors_batch(stock_codes: List[str], trade_date: date) -> Dict[str, Dict]:
    """
    批量获取筹码因子数据
    Args:
        stock_codes: 股票代码列表
        trade_date: 交易日期
    Returns:
        Dict[str, Dict]: 股票代码到因子字典的映射
    """
    result = {}
    # 按市场分组
    market_groups = {}
    for code in stock_codes:
        model = get_chip_factor_model_by_code(code)
        market_groups.setdefault(model, []).append(code)
    # 并行查询不同市场的数据
    for model, codes in market_groups.items():
        queryset = model.objects.filter(
            stock__stock_code__in=codes,
            trade_time=trade_date,
            calc_status='success'
        ).select_related('stock')
        async for factor in queryset:
            result[factor.stock.stock_code] = {
                'price_to_weight_avg_ratio': factor.price_to_weight_avg_ratio,
                'chip_concentration_ratio': factor.chip_concentration_ratio,
                'chip_stability': factor.chip_stability,
                'profit_pressure': factor.profit_pressure,
                'chip_entropy': factor.chip_entropy,
                'chip_skewness': factor.chip_skewness,
                'chip_kurtosis': factor.chip_kurtosis,
                'winner_rate': factor.winner_rate,
                'close': factor.close,
                'weight_avg_cost': factor.weight_avg_cost,
                'trade_time': factor.trade_time
            }
    return result

def get_fundflow_factors_batch(stock_codes: List[str], trade_date: date) -> Dict[str, Dict]:
    """
    批量获取资金流向因子数据
    Args:
        stock_codes: 股票代码列表
        trade_date: 交易日期
    Returns:
        Dict[str, Dict]: 股票代码到因子字典的映射
    """
    result = {}
    # 按市场分组
    market_groups = {}
    for code in stock_codes:
        model = get_fundflow_factor_model_by_code(code)
        if model:
            market_groups.setdefault(model, []).append(code)
    
    # 并行查询不同市场的数据
    for model, codes in market_groups.items():
        queryset = model.objects.filter(
            stock__stock_code__in=codes,
            trade_time=trade_date
        ).select_related('stock')
        
        for factor in queryset:
            result[factor.stock.stock_code] = {
                # 绝对量级指标
                'total_net_amount_5d': float(factor.total_net_amount_5d) if factor.total_net_amount_5d else None,
                'total_net_amount_10d': float(factor.total_net_amount_10d) if factor.total_net_amount_10d else None,
                'avg_daily_net_5d': float(factor.avg_daily_net_5d) if factor.avg_daily_net_5d else None,
                'avg_daily_net_10d': float(factor.avg_daily_net_10d) if factor.avg_daily_net_10d else None,
                
                # 相对强度指标
                'net_amount_ratio': float(factor.net_amount_ratio) if factor.net_amount_ratio else None,
                'net_amount_ratio_ma5': float(factor.net_amount_ratio_ma5) if factor.net_amount_ratio_ma5 else None,
                'flow_intensity': float(factor.flow_intensity) if factor.flow_intensity else None,
                'intensity_level': factor.intensity_level,
                
                # 主力行为模式
                'behavior_pattern': factor.behavior_pattern,
                'pattern_confidence': float(factor.pattern_confidence) if factor.pattern_confidence else None,
                'accumulation_score': float(factor.accumulation_score) if factor.accumulation_score else None,
                'pushing_score': float(factor.pushing_score) if factor.pushing_score else None,
                'distribution_score': float(factor.distribution_score) if factor.distribution_score else None,
                'shakeout_score': float(factor.shakeout_score) if factor.shakeout_score else None,
                
                # 资金流向质量评估
                'outflow_quality': float(factor.outflow_quality) if factor.outflow_quality else None,
                'inflow_persistence': factor.inflow_persistence,
                'large_order_anomaly': factor.large_order_anomaly,
                'flow_consistency': float(factor.flow_consistency) if factor.flow_consistency else None,
                
                # 多周期共振指标
                'daily_weekly_sync': float(factor.daily_weekly_sync) if factor.daily_weekly_sync else None,
                'daily_monthly_sync': float(factor.daily_monthly_sync) if factor.daily_monthly_sync else None,
                'short_mid_sync': float(factor.short_mid_sync) if factor.short_mid_sync else None,
                'mid_long_sync': float(factor.mid_long_sync) if factor.mid_long_sync else None,
                
                # 趋势动量指标
                'flow_momentum_5d': float(factor.flow_momentum_5d) if factor.flow_momentum_5d else None,
                'flow_momentum_10d': float(factor.flow_momentum_10d) if factor.flow_momentum_10d else None,
                'flow_acceleration': float(factor.flow_acceleration) if factor.flow_acceleration else None,
                'uptrend_strength': float(factor.uptrend_strength) if factor.uptrend_strength else None,
                'downtrend_strength': float(factor.downtrend_strength) if factor.downtrend_strength else None,
                
                # 量价背离指标
                'price_flow_divergence': float(factor.price_flow_divergence) if factor.price_flow_divergence else None,
                'divergence_type': factor.divergence_type,
                'divergence_strength': float(factor.divergence_strength) if factor.divergence_strength else None,
                
                # 统计特征指标
                'flow_zscore': float(factor.flow_zscore) if factor.flow_zscore else None,
                'flow_percentile': float(factor.flow_percentile) if factor.flow_percentile else None,
                'flow_volatility_10d': float(factor.flow_volatility_10d) if factor.flow_volatility_10d else None,
                
                # 预测指标
                'expected_flow_next_1d': float(factor.expected_flow_next_1d) if factor.expected_flow_next_1d else None,
                'flow_forecast_confidence': float(factor.flow_forecast_confidence) if factor.flow_forecast_confidence else None,
                'uptrend_continuation_prob': float(factor.uptrend_continuation_prob) if factor.uptrend_continuation_prob else None,
                'reversal_prob': float(factor.reversal_prob) if factor.reversal_prob else None,
                
                # 复合综合指标
                'comprehensive_score': float(factor.comprehensive_score) if factor.comprehensive_score else None,
                'trading_signal': factor.trading_signal,
                'signal_strength': float(factor.signal_strength) if factor.signal_strength else None,
                
                # 时间信息
                'trade_time': factor.trade_time
            }
    
    return result











