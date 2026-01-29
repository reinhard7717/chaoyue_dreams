# tasks\fundflow_factor_tasks.py
from chaoyue_dreams.celery import app as celery_app
from celery.result import AsyncResult
from django.utils import timezone
from django.db import transaction
from django.db.models import Q
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import asyncio
from asgiref.sync import sync_to_async, async_to_sync
import logging
from typing import List, Dict, Optional, Tuple, Set
import time
from decimal import Decimal

from stock_models.index import TradeCalendar
from utils.model_helpers import (
    get_daily_data_model_by_code,
    get_fundflow_factor_model_by_code,
    get_fundflow_factors_batch,
    get_fund_flow_model_by_code,
    get_fund_flow_dc_model_by_code,
    get_fund_flow_ths_model_by_code,
)
from services.fundflow_calculator import FundFlowFactorCalculator, CalculationContext
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from utils.cache_manager import CacheManager
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockDailyBasic

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='tasks.fundflow_factor_tasks.schedule_fundflow_factors_calculation', queue="calculator")
def schedule_fundflow_factors_calculation(self, start_date_str: str = None, batch_size: int = 10, incremental: bool = True):
    """
    调度任务：调度所有股票的资金流向因子计算
    Args:
        start_date_str: 开始计算日期 (YYYY-MM-DD格式)，如果为空且incremental=True则为增量模式
        batch_size: 每批处理的股票数量
        incremental: 是否为增量模式（默认True）
    """
    stock_basic_dao = StockBasicInfoDao(CacheManager())
    try:
        # 获取所有有效的股票代码
        # 注意：get_stock_list 返回的是 StockInfo 对象列表
        all_stocks = async_to_sync(stock_basic_dao.get_stock_list)()
        if not all_stocks:
            logger.warning("未获取到任何股票信息，无法调度计算任务")
            return {
                'status': 'warning',
                'message': '未获取到股票列表',
                'total_stocks': 0
            }
        total_stocks = len(all_stocks)
        print(f"开始调度资金流向因子计算，共 {total_stocks} 只股票")
        # 分批处理股票
        for i in range(0, total_stocks, batch_size):
            batch_stocks = list(all_stocks[i:i+batch_size])
            # 为每只股票创建独立计算任务
            for stock_info in batch_stocks:
                # 修正：从 StockInfo 对象中获取 stock_code 字符串
                # 之前的代码直接使用了 stock_info 对象作为 stock_code 参数，导致序列化问题或参数错误
                stock_code = stock_info.stock_code
                # 异步执行单个股票的计算任务
                calculate_fundflow_factors_for_stock.delay(
                    stock_code=stock_code,
                    start_date_str=start_date_str,
                    incremental=incremental
                )
            print(f"已调度第 {i//batch_size + 1} 批股票，共 {len(batch_stocks)} 只")
            # 减少休眠时间，提高调度效率，但保留少量休眠以防消息队列瞬时过载
            time.sleep(0.1)
        print(f"资金流向因子计算调度完成，共调度 {total_stocks} 只股票")
        return {
            'status': 'success',
            'message': f'已调度 {total_stocks} 只股票的计算任务',
            'total_stocks': total_stocks
        }
    except Exception as e:
        logger.error(f"调度资金流向因子计算失败: {e}", exc_info=True)
        raise self.retry(exc=e)

@celery_app.task(bind=True, name='tasks.fundflow_factor_tasks.calculate_fundflow_factors_for_stock', queue="calculator")
def calculate_fundflow_factors_for_stock(self, stock_code: str, start_date_str: str = None, incremental: bool = True):
    """
    计算单只股票的资金流向因子
    Args:
        stock_code: 股票代码
        start_date_str: 开始计算日期 (YYYY-MM-DD格式 或 YYYYMMDD格式)
        incremental: 是否为增量模式
    """
    try:
        print(f"开始计算股票 {stock_code} 的资金流向因子")
        # 1. 确定计算开始日期
        start_date = None
        if start_date_str:
            try:
                # 兼容 YYYY-MM-DD 和 YYYYMMDD 格式
                if '-' in start_date_str:
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                else:
                    start_date = datetime.strptime(start_date_str, '%Y%m%d').date()
            except ValueError:
                error_msg = f"日期格式错误: {start_date_str}，请使用 YYYY-MM-DD 或 YYYYMMDD 格式"
                logger.error(error_msg)
                return {
                    'status': 'failed',
                    'message': error_msg,
                    'stock_code': stock_code
                }
        # 2. 获取资金流向因子模型
        factor_model = get_fundflow_factor_model_by_code(stock_code)
        # 3. 获取该股票的交易日历
        trade_dates = get_trade_dates_for_stock(stock_code, start_date, incremental, factor_model)
        if not trade_dates:
            print(f"股票 {stock_code} 无需计算资金流向因子")
            return {
                'status': 'success',
                'message': f'股票 {stock_code} 无需计算',
                'stock_code': stock_code,
                'calculated_dates': 0
            }
        print(f"股票 {stock_code} 需要计算 {len(trade_dates)} 个交易日的因子")
        # 4. 分批计算（每50个交易日一批）
        batch_size = 50
        total_batches = (len(trade_dates) + batch_size - 1) // batch_size
        calculated_count = 0
        failed_dates = []
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(trade_dates))
            batch_dates = trade_dates[batch_start:batch_end]
            print(f"计算股票 {stock_code} 第 {batch_idx+1}/{total_batches} 批，共 {len(batch_dates)} 个交易日")
            # 计算本批日期的因子
            batch_calculated, batch_failed = calculate_factor_batch(
                stock_code, batch_dates, factor_model
            )
            calculated_count += batch_calculated
            failed_dates.extend(batch_failed)
            # 避免数据库连接过载
            time.sleep(0.5)
        # 5. 记录计算结果
        result = {
            'status': 'success',
            'message': f'股票 {stock_code} 计算完成',
            'stock_code': stock_code,
            'calculated_dates': calculated_count,
            'failed_dates': failed_dates if failed_dates else None
        }
        if failed_dates:
            logger.warning(f"股票 {stock_code} 计算完成，但有 {len(failed_dates)} 个交易日失败")
        else:
            print(f"股票 {stock_code} 计算完成，共计算 {calculated_count} 个交易日")
        return result
    except Exception as e:
        logger.error(f"计算股票 {stock_code} 资金流向因子失败: {e}", exc_info=True)
        raise self.retry(exc=e)

def get_trade_dates_for_stock(stock_code: str, start_date: date, incremental: bool, factor_model) -> List[date]:
    """
    获取需要计算的交易日列表
    Args:
        stock_code: 股票代码
        start_date: 指定的开始日期
        incremental: 是否为增量模式
        factor_model: 资金流向因子模型
    Returns:
        需要计算的交易日列表
    """
    try:
        # 修改：使用 get_latest_n_trade_dates 获取包含今天的最新交易日 (lte)
        # 原来的 get_latest_trade_date 使用 lt，会导致无法获取当天的交易日
        latest_dates = TradeCalendar.get_latest_n_trade_dates(n=1, reference_date=timezone.now().date())
        latest_trade_date = latest_dates[0] if latest_dates else None
        if not latest_trade_date:
            logger.warning(f"无法获取股票 {stock_code} 的最新交易日")
            return []
        # 确定实际开始日期
        if incremental and start_date is None:
            # 增量模式：从已有因子数据的最新日期开始
            latest_factor = factor_model.objects.filter(
                stock__stock_code=stock_code
            ).order_by('-trade_time').first()
            if latest_factor:
                # 从已有因子的下一个交易日开始
                calc_start_date = TradeCalendar.get_next_trade_date(reference_date=latest_factor.trade_time)
            else:
                # 没有因子数据，从最早的资金流向数据开始
                calc_start_date = get_earliest_flow_date(stock_code)
        else:
            # 指定日期模式
            calc_start_date = start_date or get_earliest_flow_date(stock_code)
        if not calc_start_date:
            logger.warning(f"无法确定股票 {stock_code} 的计算开始日期")
            return []
        # 确保开始日期不晚于最新交易日
        if calc_start_date > latest_trade_date:
            print(f"股票 {stock_code} 开始日期 {calc_start_date} 晚于最新交易日 {latest_trade_date}")
            return []
        # 获取开始日期到最新交易日之间的所有交易日
        all_trade_dates = TradeCalendar.get_trade_dates_between(start_date=calc_start_date,end_date=latest_trade_date)
        
        if not all_trade_dates:
            print(f"股票 {stock_code} 在 {calc_start_date} 到 {latest_trade_date} 之间没有交易日")
            return []
        # 修改：如果是增量模式且天数较少，不进行采样，确保计算所有缺失日期
        # 原来的 [::50] 采样会导致增量更新时漏掉最近的日期（如果 gap < 50 且 index != 0）
        if incremental and len(all_trade_dates) < 50:
            sampled_dates = all_trade_dates
        else:
            # 每隔50个交易日取一个（节省计算资源，适用于全量历史计算）
            sampled_dates = all_trade_dates[::50]
        logger.debug(f"股票 {stock_code} 原始交易日: {len(all_trade_dates)}，"
                    f"采样后: {len(sampled_dates)}")
        return sampled_dates
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 交易日失败: {e}", exc_info=True)
        return []

def get_earliest_flow_date(stock_code: str) -> Optional[date]:
    """获取最早的资金流向数据日期"""
    try:
        # 尝试从基础资金流向数据获取
        base_model = get_fund_flow_model_by_code(stock_code)
        earliest_base = base_model.objects.filter(
            stock__stock_code=stock_code
        ).order_by('trade_time').first()
        if earliest_base:
            return earliest_base.trade_time
        # 如果没有基础数据，尝试其他数据源
        ths_model = get_fund_flow_ths_model_by_code(stock_code)
        earliest_ths = ths_model.objects.filter(
            stock__stock_code=stock_code
        ).order_by('trade_time').first()
        if earliest_ths:
            return earliest_ths.trade_time
        # 尝试东方财富数据
        dc_model = get_fund_flow_dc_model_by_code(stock_code)
        earliest_dc = dc_model.objects.filter(
            stock__stock_code=stock_code
        ).order_by('trade_time').first()
        if earliest_dc:
            return earliest_dc.trade_time
        return None
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 最早资金流向日期失败: {e}")
        return None

def calculate_factor_batch(stock_code: str, trade_dates: List[date], factor_model) -> Tuple[int, List[date]]:
    """
    批量计算资金流向因子
    Returns:
        (成功计算的数量, 失败的日期列表)
    """
    calculated = 0
    failed_dates = []
    for trade_date in trade_dates:
        try:
            success = calculate_single_date_factor(
                stock_code, trade_date, factor_model
            )
            if success:
                calculated += 1
            else:
                failed_dates.append(trade_date)
        except Exception as e:
            logger.error(f"计算股票 {stock_code} 在 {trade_date} 的资金流向因子失败: {e}")
            failed_dates.append(trade_date)
    return calculated, failed_dates

def calculate_single_date_factor(stock_code: str, trade_date: date, factor_model) -> bool:
    """
    计算单个日期的资金流向因子
    修改思路：
    1. stock_basic_dao.get_stock_by_code 是异步方法，直接调用返回协程对象。
    2. 使用 async_to_sync 将其转换为同步调用，确保获取到 StockInfo 实例。
    3. 这样可以避免后续访问 stock_info.stock_code 时出现 'coroutine' object has no attribute 'stock_code' 错误。
    """
    stock_basic_dao = StockBasicInfoDao(CacheManager())
    try:
        # 1. 获取股票基本信息
        # 修正：DAO方法是异步的，需使用 async_to_sync 转换
        stock_info = async_to_sync(stock_basic_dao.get_stock_by_code)(stock_code)
        if not stock_info:
            logger.warning(f"股票 {stock_code} 不存在")
            return False
        # 2. 获取历史资金流向数据（最近30天）
        historical_flow_data = get_historical_flow_data(stock_code, trade_date, days=30)
        if not historical_flow_data or len(historical_flow_data) < 10:
            logger.warning(f"股票 {stock_code} 在 {trade_date} 的历史数据不足")
            return False
        # 3. 获取当前日资金流向数据
        current_flow_data = get_current_flow_data(stock_code, trade_date)
        if not current_flow_data:
            logger.warning(f"股票 {stock_code} 在 {trade_date} 的资金流向数据缺失")
            return False
        # 4. 获取每日基本信息
        daily_basic_data = get_daily_basic_data(stock_code, trade_date)
        # 5. 获取1分钟数据（可选）
        minute_data = get_1min_data(stock_code, trade_date)
        # 6. 构建计算上下文
        context = CalculationContext(
            stock_code=stock_code,
            trade_date=trade_date,
            current_flow_data=current_flow_data,
            historical_flow_data=historical_flow_data,
            daily_basic_data=daily_basic_data,
            minute_data_1min=minute_data
        )
        # 7. 计算因子
        calculator = FundFlowFactorCalculator(context)
        all_metrics = calculator.calculate_all_metrics()
        # 8. 保存到数据库
        save_factor_to_db(stock_info, trade_date, all_metrics, factor_model)
        logger.debug(f"成功计算股票 {stock_code} 在 {trade_date} 的资金流向因子")
        return True
    except Exception as e:
        logger.error(f"计算股票 {stock_code} 在 {trade_date} 的资金流向因子失败: {e}", exc_info=True)
        return False

def get_historical_flow_data(stock_code: str, end_date: date, days: int = 30) -> List[Dict]:
    """
    获取历史资金流向数据
    Args:
        stock_code: 股票代码
        end_date: 结束日期
        days: 需要获取的天数
    Returns:
        历史资金流向数据列表（按日期升序）
    """
    try:
        # 获取结束日期之前的N个交易日
        trade_dates = TradeCalendar.get_latest_n_trade_dates(n=days,reference_date=end_date)
        if not trade_dates:
            return []
        # 获取股票信息
        stock_info = StockInfo.objects.filter(stock_code=stock_code).first()
        if not stock_info:
            return []
        historical_data = []
        for trade_date in sorted(trade_dates):  # 按日期升序
            # 获取该日期的资金流向数据
            flow_data = get_single_date_flow_data(stock_code, trade_date, stock_info)
            if flow_data:
                # 添加日期信息
                flow_data['trade_date'] = trade_date.isoformat()
                historical_data.append(flow_data)
        return historical_data
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 历史资金流向数据失败: {e}")
        return []

def get_single_date_flow_data(stock_code: str, trade_date: date, stock_info: StockInfo = None) -> Optional[Dict]:
    """获取单个日期的资金流向数据"""
    try:
        if stock_info is None:
            stock_info = StockInfo.objects.filter(stock_code=stock_code).first()
            if not stock_info:
                return None
        # 优先使用基础资金流向数据
        base_model = get_fund_flow_model_by_code(stock_code)
        base_data = base_model.objects.filter(
            stock=stock_info,
            trade_time=trade_date
        ).first()
        if base_data:
            # 转换为字典格式
            data_dict = {
                'net_mf_amount': float(base_data.net_mf_amount) if base_data.net_mf_amount else 0.0,
                'net_mf_vol': int(base_data.net_mf_vol) if base_data.net_mf_vol else 0,
                # 添加分档数据
                'buy_sm_amount': float(base_data.buy_sm_amount) if base_data.buy_sm_amount else 0.0,
                'sell_sm_amount': float(base_data.sell_sm_amount) if base_data.sell_sm_amount else 0.0,
                'buy_md_amount': float(base_data.buy_md_amount) if base_data.buy_md_amount else 0.0,
                'sell_md_amount': float(base_data.sell_md_amount) if base_data.sell_md_amount else 0.0,
                'buy_lg_amount': float(base_data.buy_lg_amount) if base_data.buy_lg_amount else 0.0,
                'sell_lg_amount': float(base_data.sell_lg_amount) if base_data.sell_lg_amount else 0.0,
                'buy_elg_amount': float(base_data.buy_elg_amount) if base_data.buy_elg_amount else 0.0,
                'sell_elg_amount': float(base_data.sell_elg_amount) if base_data.sell_elg_amount else 0.0,
            }
            return data_dict
        # 如果没有基础数据，尝试其他数据源
        # 这里可以根据需要添加THS和DC数据源的逻辑
        return None
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 在 {trade_date} 的资金流向数据失败: {e}")
        return None

def get_current_flow_data(stock_code: str, trade_date: date) -> Optional[Dict]:
    """获取当前日的资金流向数据"""
    return get_single_date_flow_data(stock_code, trade_date)

def get_daily_basic_data(stock_code: str, trade_date: date) -> Optional[Dict]:
    """获取每日基本信息"""
    stock_time_trade_dao = StockTimeTradeDAO(CacheManager())
    try:
        # 查询日线数据
        # 修正1: DAO方法是异步的，需使用 async_to_sync 转换
        # 修正2: 原 get_daily_data_by_date 方法在DAO中存在 bug (NameError)，改用 get_stocks_daily_data 获取完整模型对象
        daily_data_list = async_to_sync(stock_time_trade_dao.get_stocks_daily_data)([stock_code], trade_date)
        
        if daily_data_list:
            daily_obj = daily_data_list[0]
            def safe_float(val):
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None
                return None
            basic_dict = {
                'close': safe_float(daily_obj.close),
                'amount': safe_float(daily_obj.amount),
                'vol': safe_float(daily_obj.vol),
                # turnover_rate 可能在 daily 表也可能在 basic 表
                'turnover_rate': safe_float(getattr(daily_obj, 'turnover_rate', None)),
            }
            # 尝试获取StockDailyBasic数据
            # 修正: DAO方法是异步的，需使用 async_to_sync 转换
            basic_model = async_to_sync(stock_time_trade_dao.get_stock_daily_basic_by_date)(stock_code, trade_date)
            if basic_model:
                basic_dict.update({
                    'turnover_rate_f': safe_float(basic_model.turnover_rate_f),
                    'volume_ratio': safe_float(basic_model.volume_ratio),
                    'pe': safe_float(basic_model.pe),
                    'pe_ttm': safe_float(basic_model.pe_ttm),
                    'pb': safe_float(basic_model.pb),
                    'total_mv': safe_float(basic_model.total_mv),
                    'circ_mv': safe_float(basic_model.circ_mv),
                })
                # 如果 daily 数据中没有 turnover_rate，尝试从 basic 数据中获取
                if basic_dict['turnover_rate'] is None:
                    basic_dict['turnover_rate'] = safe_float(getattr(basic_model, 'turnover_rate', None))
            return basic_dict
        return None
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 在 {trade_date} 的每日基本信息失败: {e}")
        return None

def get_1min_data(stock_code: str, trade_date: date) -> Optional[pd.DataFrame]:
    """获取1分钟数据"""
    stock_time_trade_dao = StockTimeTradeDAO(CacheManager())
    try:
        # 修正: DAO方法是异步的，需使用 async_to_sync 转换
        df = async_to_sync(stock_time_trade_dao.get_1_min_kline_time_by_day)(stock_code, trade_date)
        if df is None:
            return None
        return df
    except Exception as e:
        logger.debug(f"获取股票 {stock_code} 在 {trade_date} 的1分钟数据失败: {e}")
        return None

def save_factor_to_db(stock_info: StockInfo, trade_date: date, 
                     metrics: Dict, factor_model):
    """保存因子到数据库"""
    try:
        with transaction.atomic():
            # 创建或更新因子记录
            factor_obj, created = factor_model.objects.update_or_create(
                stock=stock_info,
                trade_time=trade_date,
                defaults={
                    # 绝对量级指标
                    'total_net_amount_3d': _safe_decimal(metrics.get('total_net_amount_3d')),
                    'total_net_amount_5d': _safe_decimal(metrics.get('total_net_amount_5d')),
                    'total_net_amount_10d': _safe_decimal(metrics.get('total_net_amount_10d')),
                    'total_net_amount_20d': _safe_decimal(metrics.get('total_net_amount_20d')),
                    'avg_daily_net_5d': _safe_decimal(metrics.get('avg_daily_net_5d')),
                    'avg_daily_net_10d': _safe_decimal(metrics.get('avg_daily_net_10d')),
                    'avg_daily_net_20d': _safe_decimal(metrics.get('avg_daily_net_20d')),
                    'total_volume_5d': _safe_decimal(metrics.get('total_volume_5d')),
                    'total_volume_10d': _safe_decimal(metrics.get('total_volume_10d')),
                    # 相对强度指标
                    'net_amount_ratio': _safe_decimal(metrics.get('net_amount_ratio')),
                    'net_amount_ratio_ma5': _safe_decimal(metrics.get('net_amount_ratio_ma5')),
                    'net_amount_ratio_ma10': _safe_decimal(metrics.get('net_amount_ratio_ma10')),
                    'flow_intensity': _safe_decimal(metrics.get('flow_intensity')),
                    'intensity_level': metrics.get('intensity_level'),
                    # 主力行为模式识别
                    'accumulation_score': _safe_decimal(metrics.get('accumulation_score')),
                    'pushing_score': _safe_decimal(metrics.get('pushing_score')),
                    'distribution_score': _safe_decimal(metrics.get('distribution_score')),
                    'shakeout_score': _safe_decimal(metrics.get('shakeout_score')),
                    'behavior_pattern': metrics.get('behavior_pattern'),
                    'pattern_confidence': _safe_decimal(metrics.get('pattern_confidence')),
                    # 资金流向质量评估
                    'outflow_quality': _safe_decimal(metrics.get('outflow_quality')),
                    'inflow_persistence': metrics.get('inflow_persistence'),
                    'large_order_anomaly': metrics.get('large_order_anomaly'),
                    'anomaly_intensity': _safe_decimal(metrics.get('anomaly_intensity')),
                    'flow_consistency': _safe_decimal(metrics.get('flow_consistency')),
                    'flow_stability': _safe_decimal(metrics.get('flow_stability')),
                    # 多周期资金共振指标
                    'daily_weekly_sync': _safe_decimal(metrics.get('daily_weekly_sync')),
                    'daily_monthly_sync': _safe_decimal(metrics.get('daily_monthly_sync')),
                    'short_mid_sync': _safe_decimal(metrics.get('short_mid_sync')),
                    'mid_long_sync': _safe_decimal(metrics.get('mid_long_sync')),
                    # 趋势动量指标
                    'flow_momentum_5d': _safe_decimal(metrics.get('flow_momentum_5d')),
                    'flow_momentum_10d': _safe_decimal(metrics.get('flow_momentum_10d')),
                    'flow_acceleration': _safe_decimal(metrics.get('flow_acceleration')),
                    'uptrend_strength': _safe_decimal(metrics.get('uptrend_strength')),
                    'downtrend_strength': _safe_decimal(metrics.get('downtrend_strength')),
                    # 量价背离指标
                    'price_flow_divergence': _safe_decimal(metrics.get('price_flow_divergence')),
                    'divergence_type': metrics.get('divergence_type'),
                    'divergence_strength': _safe_decimal(metrics.get('divergence_strength')),
                    # 统计特征指标
                    'flow_zscore': _safe_decimal(metrics.get('flow_zscore')),
                    'flow_percentile': _safe_decimal(metrics.get('flow_percentile')),
                    'flow_volatility_10d': _safe_decimal(metrics.get('flow_volatility_10d')),
                    'flow_volatility_20d': _safe_decimal(metrics.get('flow_volatility_20d')),
                    # 预测指标
                    'expected_flow_next_1d': _safe_decimal(metrics.get('expected_flow_next_1d')),
                    'flow_forecast_confidence': _safe_decimal(metrics.get('flow_forecast_confidence')),
                    'uptrend_continuation_prob': _safe_decimal(metrics.get('uptrend_continuation_prob')),
                    'reversal_prob': _safe_decimal(metrics.get('reversal_prob')),
                    # 复合综合指标
                    'comprehensive_score': _safe_decimal(metrics.get('comprehensive_score')),
                    'trading_signal': metrics.get('trading_signal'),
                    'signal_strength': _safe_decimal(metrics.get('signal_strength')),
                    # 原始数据快照
                    'flow_sequence_30d': metrics.get('flow_sequence_30d'),
                    'feature_vector': metrics.get('feature_vector'),
                    'calculation_metadata': metrics.get('calculation_metadata'),
                }
            )
            logger.debug(f"{'创建' if created else '更新'}股票 {stock_info.stock_code} "
                        f"在 {trade_date} 的资金流向因子")
            return factor_obj
            
    except Exception as e:
        logger.error(f"保存股票 {stock_info.stock_code} 在 {trade_date} 的资金流向因子失败: {e}")
        raise

def _safe_decimal(value):
    """安全转换为Decimal"""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except:
        return None

@celery_app.task(bind=True, queue="calculator")
def update_fundflow_factors_daily(self):
    """
    每日更新任务：更新最新交易日的资金流向因子
    """
    stock_basic_dao = StockBasicInfoDao(CacheManager())
    try:
        # 获取最新的交易日
        # 修改：使用 get_latest_n_trade_dates 确保包含当天
        latest_dates = TradeCalendar.get_latest_n_trade_dates(n=1,reference_date=timezone.now().date())
        latest_trade_date = latest_dates[0] if latest_dates else None
        if not latest_trade_date:
            logger.warning("无法获取最新交易日")
            return {'status': 'failed', 'message': '无法获取最新交易日'}
        print(f"开始更新 {latest_trade_date} 的资金流向因子")
        # 获取所有有效的股票代码
        all_stocks = async_to_sync(stock_basic_dao.get_stock_list)()
        total_stocks = len(all_stocks)
        print(f"需要更新 {total_stocks} 只股票的资金流向因子")
        # 批量处理股票
        batch_size = 50
        successful = 0
        failed = []
        for i in range(0, total_stocks, batch_size):
            batch_stocks = list(all_stocks[i:i+batch_size])
            for stock_code in batch_stocks:
                try:
                    factor_model = get_fundflow_factor_model_by_code(stock_code)
                    # 检查是否已计算
                    existing = factor_model.objects.filter(
                        stock__stock_code=stock_code,
                        trade_time=latest_trade_date
                    ).exists()
                    if existing:
                        logger.debug(f"股票 {stock_code} 在 {latest_trade_date} 的因子已存在，跳过")
                        successful += 1
                        continue
                    # 计算因子
                    success = calculate_single_date_factor(
                        stock_code, latest_trade_date, factor_model
                    )
                    if success:
                        successful += 1
                    else:
                        failed.append(stock_code)
                except Exception as e:
                    logger.error(f"更新股票 {stock_code} 资金流向因子失败: {e}")
                    failed.append(stock_code)
            print(f"已更新第 {i//batch_size + 1} 批股票，成功: {successful}, 失败: {len(failed)}")
            time.sleep(0.5)  # 避免过载
        result = {
            'status': 'success',
            'message': f'完成每日资金流向因子更新',
            'date': latest_trade_date.isoformat(),
            'total_stocks': total_stocks,
            'successful': successful,
            'failed': len(failed),
            'failed_stocks': failed if failed else None
        }
        print(f"每日资金流向因子更新完成: {result}")
        return result
    except Exception as e:
        logger.error(f"每日更新资金流向因子失败: {e}", exc_info=True)
        raise self.retry(exc=e)

@celery_app.task(bind=True, queue="calculator")
def test_fundflow_factor_calculation(self, stock_code: str = '000001.SZ', test_date_str: str = None):
    """
    测试任务：测试单只股票单日的资金流向因子计算
    Args:
        stock_code: 测试股票代码
        test_date_str: 测试日期 (YYYY-MM-DD格式)
    """
    try:
        if test_date_str:
            test_date = datetime.strptime(test_date_str, '%Y-%m-%d').date()
        else:
            # 使用最近的一个交易日
            # 修改：使用 get_latest_n_trade_dates 确保包含当天
            latest_dates = TradeCalendar.get_latest_n_trade_dates(n=1,reference_date=timezone.now().date())
            test_date = latest_dates[0] if latest_dates else None
        if not test_date:
            return {'status': 'failed', 'message': '无法获取测试日期'}
        print(f"测试股票 {stock_code} 在 {test_date} 的资金流向因子计算")
        # 获取因子模型
        factor_model = get_fundflow_factor_model_by_code(stock_code)
        # 计算因子
        success = calculate_single_date_factor(
            stock_code, test_date, factor_model
        )
        result = {
            'status': 'success' if success else 'failed',
            'stock_code': stock_code,
            'date': test_date.isoformat(),
            'calculated': success
        }
        return result
    except Exception as e:
        logger.error(f"测试资金流向因子计算失败: {e}", exc_info=True)
        return {'status': 'failed', 'message': str(e)}












