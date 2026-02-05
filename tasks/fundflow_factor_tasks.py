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
from typing import List, Dict, Optional, Tuple, Set, Any
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
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
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
    计算单只股票的资金流向因子 (入口包装器)
    版本: V1.5
    说明: 使用 async_to_sync 包裹整个异步处理流程，确保单任务单循环，解决 Redis 连接泄露问题。
    """
    try:
        # 将异步逻辑同步化执行
        return async_to_sync(_process_stock_factors_async)(stock_code, start_date_str, incremental)
    except Exception as e:
        logger.error(f"计算股票 {stock_code} 资金流向因子失败: {e}", exc_info=True)
        raise self.retry(exc=e)

async def _process_stock_factors_async(stock_code: str, start_date_str: str = None, incremental: bool = True):
    """
    [Async] 核心处理逻辑
    版本: V1.6
    说明: 引入 StockRealtimeDAO 并传递给计算流程，确保Tick数据可获取。
    """
    # 初始化共享的 CacheManager 和 DAO (在当前事件循环中)
    cache_mgr = CacheManager()
    stock_basic_dao = StockBasicInfoDao(cache_mgr)
    stock_time_trade_dao = StockTimeTradeDAO(cache_mgr)
    
    # [新增] 初始化 StockRealtimeDAO 用于获取Tick数据
    realtime_dao = StockRealtimeDAO(cache_mgr)

    try:
        # 1. 确定计算开始日期
        start_date = None
        if start_date_str:
            try:
                if '-' in start_date_str:
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                else:
                    start_date = datetime.strptime(start_date_str, '%Y%m%d').date()
            except ValueError:
                return {'status': 'failed', 'message': f"日期格式错误: {start_date_str}"}
        # 2. 获取模型和交易日历 (涉及 DB 操作，需 sync_to_async)
        factor_model = await sync_to_async(get_fundflow_factor_model_by_code)(stock_code)
        # 获取交易日历
        trade_dates = await sync_to_async(get_trade_dates_for_stock)(
            stock_code, start_date, incremental, factor_model
        )
        if not trade_dates:
            return {
                'status': 'success', 
                'message': f'股票 {stock_code} 无需计算',
                'calculated_dates': 0
            }
        # 3. 分批计算
        batch_size = 50
        total_batches = (len(trade_dates) + batch_size - 1) // batch_size
        calculated_count = 0
        failed_dates = []
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(trade_dates))
            batch_dates = trade_dates[batch_start:batch_end]
            # 调用异步批量计算，传入 realtime_dao
            batch_calculated, batch_failed = await _calculate_factor_batch_async(
                stock_code, batch_dates, factor_model,
                stock_basic_dao, stock_time_trade_dao, realtime_dao
            )
            calculated_count += batch_calculated
            failed_dates.extend(batch_failed)
            # 异步休眠
            await asyncio.sleep(0.5)
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
        
    finally:
        # 尝试关闭 CacheManager 连接 (如果支持)
        if hasattr(cache_mgr, 'close'):
            await cache_mgr.close()

async def _calculate_factor_batch_async(stock_code: str, trade_dates: List[date], factor_model, stock_basic_dao, stock_time_trade_dao, realtime_dao) -> Tuple[int, List[date]]:
    """
    v1.3.0: 升级为累积实例后批量入库模式。
    通过 _calculate_single_date_factor_async 获取实例，由 save_factors_bulk 统一提交。
    """
    failed_dates = []
    instances_to_save = []
    for trade_date in trade_dates:
        try:
            # 修改单日计算方法，使其返回模型实例
            instance = await _calculate_single_date_factor_async(
                stock_code, trade_date, factor_model,
                stock_basic_dao, stock_time_trade_dao, realtime_dao
            )
            if instance:
                instances_to_save.append(instance)
            else:
                failed_dates.append(trade_date)
        except Exception as e:
            logger.error(f"准备股票 {stock_code} 在 {trade_date} 的因子实例失败: {e}")
            failed_dates.append(trade_date)
    # 批量入库
    if instances_to_save:
        try:
            await sync_to_async(save_factors_bulk)(factor_model, instances_to_save)
        except Exception as e:
            logger.error(f"股票 {stock_code} 批量入库失败: {e}")
            return 0, trade_dates
    return len(instances_to_save), failed_dates

async def _calculate_single_date_factor_async(stock_code: str, trade_date: date, factor_model, stock_basic_dao, stock_time_trade_dao, realtime_dao) -> Optional[Any]:
    """
    v1.7.1: 增加动态字段过滤逻辑，修复计算指标与模型字段不一致导致的实例化失败。
    通过 _meta.fields 提取合法字段名，确保 bulk_create 实例化的健壮性。
    """
    try:
        stock_info = await stock_basic_dao.get_stock_by_code(stock_code)
        if not stock_info: return None
        historical_flow_data = await _get_historical_flow_data_async(stock_code, trade_date, stock_time_trade_dao, days=120)
        if not historical_flow_data or len(historical_flow_data) < 10: return None
        current_flow_data = await sync_to_async(get_current_flow_data)(stock_code, trade_date)
        if not current_flow_data: return None
        daily_basic_data = await _get_daily_basic_data_async(stock_code, trade_date, stock_time_trade_dao)
        minute_data = await _get_1min_data_async(stock_code, trade_date, stock_time_trade_dao)
        tick_data = await realtime_dao.get_daily_real_ticks(stock_code, trade_date) if realtime_dao else None
        context = CalculationContext(
            stock_code=stock_code, trade_date=trade_date, current_flow_data=current_flow_data,
            historical_flow_data=historical_flow_data, daily_basic_data=daily_basic_data,
            minute_data_1min=minute_data, tick_data=tick_data
        )
        calculator = FundFlowFactorCalculator(context)
        metrics = calculator.calculate_all_metrics()
        # 动态过滤：仅保留数据库中存在的字段
        model_fields = {f.name for f in factor_model._meta.fields}
        # 预定义非数值字段，避免经过 _safe_decimal 转换
        raw_fields = {'behavior_pattern', 'divergence_type', 'trading_signal', 'feature_vector', 'intraday_flow_distribution'}
        final_data = {}
        for key, value in metrics.items():
            if key in model_fields:
                if key in raw_fields:
                    final_data[key] = value
                else:
                    final_data[key] = _safe_decimal(value)
        # 补充关联键
        final_data['stock'] = stock_info
        final_data['trade_time'] = trade_date
        return factor_model(**final_data)
    except Exception as e:
        logger.error(f"构建因子实例失败 {stock_code} @ {trade_date}: {e}")
        return None

async def _get_historical_flow_data_async(stock_code: str, end_date: date, stock_time_trade_dao, days: int = 120) -> List[Dict]:
    """
    [Async] 获取历史资金流向数据
    版本: V1.8
    修改思路:
    1. 增加 'vol' 字段的提取和合并，解决成交量指标为0的问题。
    """
    try:
        # DB 操作
        trade_dates = await sync_to_async(TradeCalendar.get_latest_n_trade_dates)(n=days, reference_date=end_date)
        if not trade_dates:
            return []
        stock_info = await sync_to_async(StockInfo.objects.filter(stock_code=stock_code).first)()
        if not stock_info:
            return []
        historical_data = []
        # 循环获取单日流向数据
        sorted_dates = sorted(trade_dates)
        for trade_date in sorted_dates:
            flow_data = await sync_to_async(get_single_date_flow_data)(stock_code, trade_date, stock_info)
            if flow_data:
                flow_data['trade_date'] = trade_date.isoformat()
                flow_data['net_amount_ratio'] = 0.0
                historical_data.append(flow_data)
        if not historical_data:
            return []
        # 异步获取历史行情
        try:
            start_date = sorted_dates[0]
            real_end_date = sorted_dates[-1]
            s_str = start_date.strftime('%Y%m%d')
            e_str = real_end_date.strftime('%Y%m%d')
            # 直接 await DAO 方法
            df_price = await stock_time_trade_dao.get_daily_data(stock_code, s_str, e_str)
            if not df_price.empty:
                price_map = {}
                for idx, row in df_price.iterrows():
                    try:
                        if hasattr(idx, 'strftime'):
                            d_str = idx.strftime('%Y-%m-%d')
                        else:
                            d_str = pd.to_datetime(idx).strftime('%Y-%m-%d')
                    except Exception:
                        continue
                    # [关键修正] 提取 vol 字段
                    price_map[d_str] = {
                        'close': float(row['close']) if pd.notnull(row.get('close')) else None,
                        'pct_change': float(row['pct_change']) if pd.notnull(row.get('pct_change')) else None,
                        'amount': float(row['amount']) if pd.notnull(row.get('amount')) else 0.0,
                        'vol': float(row['vol']) if pd.notnull(row.get('vol')) else 0.0
                    }
                for item in historical_data:
                    d_str = item['trade_date']
                    if d_str in price_map:
                        item.update(price_map[d_str])
                    net_amt = item.get('net_mf_amount')
                    total_amt = item.get('amount')
                    if net_amt is not None and total_amt and total_amt != 0:
                        item['net_amount_ratio'] = (net_amt / total_amt) * 1000.0
                    else:
                        item['net_amount_ratio'] = 0.0
                        
        except Exception as e:
            logger.error(f"合并股票 {stock_code} 历史行情数据失败: {e}", exc_info=True)
        return historical_data
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 历史资金流向数据失败: {e}")
        return []

async def _get_daily_basic_data_async(stock_code: str, trade_date: date, 
                                    stock_time_trade_dao) -> Optional[Dict]:
    """[Async] 获取每日基本信息"""
    try:
        # 直接 await DAO 方法
        daily_data_list = await stock_time_trade_dao.get_stocks_daily_data([stock_code], trade_date)
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
                'turnover_rate': safe_float(getattr(daily_obj, 'turnover_rate', None)),
            }
            basic_model = await stock_time_trade_dao.get_stock_daily_basic_by_date(stock_code, trade_date)
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
                if basic_dict['turnover_rate'] is None:
                    basic_dict['turnover_rate'] = safe_float(getattr(basic_model, 'turnover_rate', None))
            return basic_dict
        return None
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 在 {trade_date} 的每日基本信息失败: {e}")
        return None

def get_trade_dates_for_stock(stock_code: str, start_date: date, incremental: bool, factor_model) -> List[date]:
    """
    获取需要计算的交易日列表
    版本: V1.2
    说明: 移除错误的采样逻辑 [::50]，确保计算指定区间内的所有交易日。
    """
    try:
        # 修改：使用 get_latest_n_trade_dates 获取包含今天的最新交易日 (lte)
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
        # [修正] 移除采样逻辑，返回所有需要计算的日期
        # 原代码的 [::50] 会导致只计算极少数日期，不符合连续因子计算的需求
        logger.debug(f"股票 {stock_code} 需要计算的交易日数量: {len(all_trade_dates)}")
        return all_trade_dates
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

def calculate_factor_batch(stock_code: str, trade_dates: List[date], factor_model, stock_basic_dao, stock_time_trade_dao) -> Tuple[int, List[date]]:
    """
    v1.2.0: 同步版批量计算优化。
    采用列表推导式收集实例并调用批量保存。
    """
    instances = []
    failed_dates = []
    for trade_date in trade_dates:
        try:
            # 调用修改后的同步单日计算方法
            instance = calculate_single_date_factor(stock_code, trade_date, factor_model, stock_basic_dao, stock_time_trade_dao)
            if instance:
                instances.append(instance)
            else:
                failed_dates.append(trade_date)
        except Exception as e:
            failed_dates.append(trade_date)
    if instances:
        save_factors_bulk(factor_model, instances)
    return len(instances), failed_dates

def calculate_single_date_factor(stock_code: str, trade_date: date, factor_model, stock_basic_dao, stock_time_trade_dao) -> Optional[Any]:
    """
    v1.7.1: 同步版增加动态字段过滤，确保 metrics 键值对与 factor_model 字段定义完全匹配。
    """
    try:
        stock_info = async_to_sync(stock_basic_dao.get_stock_by_code)(stock_code)
        if not stock_info: return None
        historical_flow_data = get_historical_flow_data(stock_code, trade_date, stock_time_trade_dao, stock_info, days=120)
        if not historical_flow_data or len(historical_flow_data) < 10: return None
        current_flow_data = get_current_flow_data(stock_code, trade_date)
        if not current_flow_data: return None
        daily_basic_data = get_daily_basic_data(stock_code, trade_date, stock_time_trade_dao)
        minute_data = get_1min_data(stock_code, trade_date, stock_time_trade_dao)
        from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
        realtime_dao = StockRealtimeDAO(stock_basic_dao.cache_manager)
        tick_data = async_to_sync(realtime_dao.get_daily_real_ticks)(stock_code, trade_date)
        context = CalculationContext(
            stock_code=stock_code, trade_date=trade_date, current_flow_data=current_flow_data,
            historical_flow_data=historical_flow_data, daily_basic_data=daily_basic_data,
            minute_data_1min=minute_data, tick_data=tick_data
        )
        calculator = FundFlowFactorCalculator(context)
        metrics = calculator.calculate_all_metrics()
        model_fields = {f.name for f in factor_model._meta.fields}
        raw_fields = {'behavior_pattern', 'divergence_type', 'trading_signal', 'feature_vector', 'intraday_flow_distribution'}
        final_data = {}
        for key, value in metrics.items():
            if key in model_fields:
                if key in raw_fields:
                    final_data[key] = value
                else:
                    final_data[key] = _safe_decimal(value)
        final_data['stock'] = stock_info
        final_data['trade_time'] = trade_date
        return factor_model(**final_data)
    except Exception as e:
        logger.error(f"同步构建因子实例失败 {stock_code} @ {trade_date}: {e}")
        return None

def get_historical_flow_data(stock_code: str, end_date: date, 
                            stock_time_trade_dao: StockTimeTradeDAO, stock_info: StockInfo,
                            days: int = 120) -> List[Dict]:
    """
    获取历史资金流向数据
    版本: V1.7
    说明: 
    1. 接收 stock_time_trade_dao 参数，复用 Redis 连接。
    2. 移除内部的 CacheManager 实例化。
    """
    try:
        # 获取结束日期之前的N个交易日
        trade_dates = TradeCalendar.get_latest_n_trade_dates(n=days, reference_date=end_date)
        if not trade_dates:
            return []
        if not stock_info:
            return []
        historical_data = []
        # 1. 获取资金流向数据
        for trade_date in sorted(trade_dates):
            flow_data = get_single_date_flow_data(stock_code, trade_date, stock_info)
            if flow_data:
                flow_data['trade_date'] = trade_date.isoformat()
                flow_data['net_amount_ratio'] = 0.0
                historical_data.append(flow_data)
        if not historical_data:
            return []
        # 2. 批量获取历史行情数据 (Close, PctChange, Amount) 并合并
        try:
            sorted_dates = sorted(trade_dates)
            start_date = sorted_dates[0]
            real_end_date = sorted_dates[-1]
            # 使用传入的 stock_time_trade_dao
            s_str = start_date.strftime('%Y%m%d')
            e_str = real_end_date.strftime('%Y%m%d')
            # 异步转同步调用 DAO 获取日线数据
            df_price = async_to_sync(stock_time_trade_dao.get_daily_data)(stock_code, s_str, e_str)
            if not df_price.empty:
                # 构建价格查找字典
                price_map = {}
                for idx, row in df_price.iterrows():
                    try:
                        if hasattr(idx, 'strftime'):
                            d_str = idx.strftime('%Y-%m-%d')
                        else:
                            d_str = pd.to_datetime(idx).strftime('%Y-%m-%d')
                    except Exception:
                        continue
                    price_map[d_str] = {
                        'close': float(row['close']) if pd.notnull(row.get('close')) else None,
                        'pct_change': float(row['pct_change']) if pd.notnull(row.get('pct_change')) else None,
                        'amount': float(row['amount']) if pd.notnull(row.get('amount')) else 0.0
                    }
                # 将价格数据合并到 historical_data 中
                for item in historical_data:
                    d_str = item['trade_date']
                    if d_str in price_map:
                        item.update(price_map[d_str])
                    # 计算 net_amount_ratio
                    net_amt = item.get('net_mf_amount')
                    total_amt = item.get('amount')
                    if net_amt is not None and total_amt and total_amt != 0:
                        item['net_amount_ratio'] = (net_amt / total_amt) * 1000.0
                    else:
                        item['net_amount_ratio'] = 0.0
            else:
                logger.warning(f"股票 {stock_code} 在 {s_str}-{e_str} 期间无日线行情数据，无法计算净流入占比")
        except Exception as e:
            logger.error(f"合并股票 {stock_code} 历史行情数据失败: {e}", exc_info=True)
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

def get_daily_basic_data(stock_code: str, trade_date: date, 
                        stock_time_trade_dao: StockTimeTradeDAO) -> Optional[Dict]:
    """
    获取每日基本信息
    版本: V1.1
    说明: 接收 stock_time_trade_dao 参数，复用 Redis 连接。
    """
    try:
        # 使用传入的 stock_time_trade_dao
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
                'turnover_rate': safe_float(getattr(daily_obj, 'turnover_rate', None)),
            }
            # 尝试获取StockDailyBasic数据
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
                if basic_dict['turnover_rate'] is None:
                    basic_dict['turnover_rate'] = safe_float(getattr(basic_model, 'turnover_rate', None))
            return basic_dict
        return None
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 在 {trade_date} 的每日基本信息失败: {e}")
        return None

async def _get_1min_data_async(stock_code: str, trade_date: date, stock_time_trade_dao) -> Optional[pd.DataFrame]:
    """[Async] 获取1分钟数据"""
    try:
        df = await stock_time_trade_dao.get_1_min_kline_time_by_day(stock_code, trade_date)
        return df
    except Exception as e:
        logger.debug(f"获取股票 {stock_code} 在 {trade_date} 的1分钟数据失败: {e}")
        return None

def get_1min_data(stock_code: str, trade_date: date, stock_time_trade_dao: StockTimeTradeDAO) -> Optional[pd.DataFrame]:
    """
    获取1分钟数据
    版本: V1.1
    说明: 接收 stock_time_trade_dao 参数，复用 Redis 连接。
    """
    try:
        # 使用传入的 stock_time_trade_dao
        df = async_to_sync(stock_time_trade_dao.get_1_min_kline_time_by_day)(stock_code, trade_date)
        if df is None:
            return None
        return df
    except Exception as e:
        logger.debug(f"获取股票 {stock_code} 在 {trade_date} 的1分钟数据失败: {e}")
        return None

def save_factor_to_db(stock_info: StockInfo, trade_date: date, metrics: Dict, factor_model):
    """
    v1.4.1: 增强型入库逻辑。
    引入 select_for_update 配合 atomic 降低死锁概率，并对 Decimal 转换进行极致容错。
    """
    from django.db import transaction
    from django.db.utils import OperationalError
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with transaction.atomic():
                # 先尝试锁定行，减少 update_or_create 内部竞争导致的死锁
                factor_model.objects.select_for_update().filter(
                    stock=stock_info, trade_time=trade_date
                ).first()
                # 数据清理：确保所有存入 Decimal 的值不是 NaN 或 Inf
                defaults = {k: _safe_decimal(v) for k, v in metrics.items() if not isinstance(v, (str, bool, type(None)))}
                # 补充非数值字段
                defaults.update({
                    'behavior_pattern': metrics.get('behavior_pattern'),
                    'divergence_type': metrics.get('divergence_type'),
                    'trading_signal': metrics.get('trading_signal'),
                    'feature_vector': metrics.get('feature_vector'),
                    'intraday_flow_distribution': metrics.get('intraday_flow_distribution'),
                })
                factor_obj, created = factor_model.objects.update_or_create(
                    stock=stock_info, trade_time=trade_date, defaults=defaults
                )
                return factor_obj
        except OperationalError as e:
            if e.args[0] == 1213 and attempt < max_retries - 1:
                time.sleep(0.2 * (attempt + 1))
                continue
            raise

def save_factors_bulk(factor_model, objects_to_save: List[Any]):
    """
    v2.0.1: 针对 MySQL 修正批量入库逻辑。
    移除 unique_fields 参数，因为 MySQL 的 ON DUPLICATE KEY UPDATE 机制
    不需要显式指定冲突字段（自动匹配所有唯一索引）。
    """
    if not objects_to_save:
        return
    from django.db import transaction
    from django.db.utils import OperationalError
    import time

    # 获取所有非主键和非关联键的待更新字段
    # 排除 'id', 'stock', 'trade_time' (作为唯一索引项，MySQL中不能更新唯一键本身)
    # 同时排除 'created_at'，保留 'updated_at' (虽然 auto_now 会处理，但显式包含更稳妥，或者让 DB 处理)
    # 注意：bulk_create 不会触发 auto_now，所以如果有 updated_at 需手动更新或包含在内
    all_fields = [f.name for f in factor_model._meta.fields]
    exclude_fields = {'id', 'stock', 'trade_time', 'created_at'}
    update_fields = [f for f in all_fields if f not in exclude_fields]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with transaction.atomic():
                # MySQL 特有修正：不传递 unique_fields
                factor_model.objects.bulk_create(
                    objects_to_save,
                    batch_size=500,
                    update_conflicts=True,
                    update_fields=update_fields
                    # unique_fields=['stock', 'trade_time']  <-- 已移除，MySQL 不支持此参数
                )
            break
        except OperationalError as e:
            # 1213 是 MySQL 死锁错误码
            if e.args[0] == 1213 and attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
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
    每日更新任务 (入口包装器)
    """
    try:
        return async_to_sync(_update_fundflow_factors_daily_async)()
    except Exception as e:
        logger.error(f"每日更新资金流向因子失败: {e}", exc_info=True)
        raise self.retry(exc=e)

async def _update_fundflow_factors_daily_async():
    """
    [Async] 每日更新逻辑
    版本: V1.2
    说明: 引入 StockRealtimeDAO。
    """
    cache_mgr = CacheManager()
    stock_basic_dao = StockBasicInfoDao(cache_mgr)
    stock_time_trade_dao = StockTimeTradeDAO(cache_mgr)
    
    # [新增] 初始化 StockRealtimeDAO
    from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
    realtime_dao = StockRealtimeDAO(cache_mgr)

    try:
        latest_dates = await sync_to_async(TradeCalendar.get_latest_n_trade_dates)(n=1, reference_date=timezone.now().date())
        latest_trade_date = latest_dates[0] if latest_dates else None
        if not latest_trade_date:
            return {'status': 'failed', 'message': '无法获取最新交易日'}
        all_stocks = await stock_basic_dao.get_stock_list()
        total_stocks = len(all_stocks)
        batch_size = 50
        successful = 0
        failed = []
        for i in range(0, total_stocks, batch_size):
            batch_stocks = list(all_stocks[i:i+batch_size])
            for stock_code in batch_stocks:
                try:
                    factor_model = await sync_to_async(get_fundflow_factor_model_by_code)(stock_code)
                    existing = await sync_to_async(factor_model.objects.filter(
                        stock__stock_code=stock_code,
                        trade_time=latest_trade_date
                    ).exists)()
                    if existing:
                        successful += 1
                        continue
                    # 传入 realtime_dao
                    success = await _calculate_single_date_factor_async(
                        stock_code, latest_trade_date, factor_model,
                        stock_basic_dao, stock_time_trade_dao, realtime_dao
                    )
                    if success:
                        successful += 1
                    else:
                        failed.append(stock_code)
                except Exception as e:
                    failed.append(stock_code)
            await asyncio.sleep(0.5)
        return {
            'status': 'success',
            'date': latest_trade_date.isoformat(),
            'successful': successful,
            'failed': len(failed)
        }
    finally:
        if hasattr(cache_mgr, 'close'):
            await cache_mgr.close()

@celery_app.task(bind=True, queue="calculator")
def test_fundflow_factor_calculation(self, stock_code: str = '000001.SZ', test_date_str: str = None):
    """测试任务 (入口包装器)"""
    return async_to_sync(_test_fundflow_factor_calculation_async)(stock_code, test_date_str)

async def _test_fundflow_factor_calculation_async(stock_code: str, test_date_str: str = None):
    """
    [Async] 测试逻辑
    版本: V1.2
    说明: 引入 StockRealtimeDAO。
    """
    cache_mgr = CacheManager()
    stock_basic_dao = StockBasicInfoDao(cache_mgr)
    stock_time_trade_dao = StockTimeTradeDAO(cache_mgr)    
    realtime_dao = StockRealtimeDAO(cache_mgr)

    try:
        if test_date_str:
            test_date = datetime.strptime(test_date_str, '%Y-%m-%d').date()
        else:
            latest_dates = await sync_to_async(TradeCalendar.get_latest_n_trade_dates)(n=1, reference_date=timezone.now().date())
            test_date = latest_dates[0] if latest_dates else None
        if not test_date:
            return {'status': 'failed', 'message': '无法获取测试日期'}
        factor_model = await sync_to_async(get_fundflow_factor_model_by_code)(stock_code)
        # 传入 realtime_dao
        success = await _calculate_single_date_factor_async(
            stock_code, test_date, factor_model,
            stock_basic_dao, stock_time_trade_dao, realtime_dao
        )
        return {
            'status': 'success' if success else 'failed',
            'stock_code': stock_code,
            'date': test_date.isoformat()
        }
    finally:
        if hasattr(cache_mgr, 'close'):
            await cache_mgr.close()












