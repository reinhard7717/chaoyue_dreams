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
def schedule_fundflow_factors_calculation(self, start_date_str: str = None, batch_size: int = 50, incremental: bool = True):
    """
    版本: V2.0.0
    说明: 重构调度任务，引入增量更新前置过滤机制(Pre-dispatch Filtering)。
    利用 asyncio 并发探测各动态分表模型的最新计算日期，仅将实际落后于最新交易日的股票压入计算队列，
    极大降低 Celery Broker 队列压力与无意义的 Worker 数据库建连及唤醒开销。
    """
    from utils.cache_manager import CacheManager
    from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
    from stock_models.index import TradeCalendar
    from django.db.models import Max
    from datetime import date
    import asyncio
    from asgiref.sync import async_to_sync, sync_to_async
    stock_basic_dao = StockBasicInfoDao(CacheManager())
    try:
        all_stocks = async_to_sync(stock_basic_dao.get_stock_list)()
        if not all_stocks:
            logger.warning("未获取到任何股票信息，无法调度计算任务")
            return {'status': 'warning', 'message': '未获取到股票列表', 'total_stocks': 0}
        total_stocks = len(all_stocks)
        print(f"开始调度资金流向因子计算，共获取 {total_stocks} 只股票")
        global_latest_date = None
        if incremental and not start_date_str:
            latest_dates = TradeCalendar.get_latest_n_trade_dates(n=1, reference_date=timezone.now().date())
            global_latest_date = latest_dates[0] if latest_dates else None
            if not global_latest_date:
                logger.warning("增量模式下无法获取全局最新交易日，调度中止")
                return {'status': 'failed', 'message': '无法获取最新交易日'}
        async def _check_stock_needs_update(stock_code: str, target_date: date) -> bool:
            try:
                factor_model = await sync_to_async(get_fundflow_factor_model_by_code)(stock_code)
                latest_record = await sync_to_async(lambda: factor_model.objects.filter(stock__stock_code=stock_code).aggregate(max_date=Max('trade_time')))()
                max_date = latest_record.get('max_date')
                if max_date is None or max_date < target_date:
                    return True
                return False
            except Exception as e:
                logger.error(f"探测股票 {stock_code} 最新计算日期异常: {e}")
                return True
        async def _filter_batch_stocks(stocks: list) -> list:
            tasks = []
            for stock_info in stocks:
                stock_code = stock_info.stock_code
                if global_latest_date:
                    tasks.append(_check_stock_needs_update(stock_code, global_latest_date))
                else:
                    async def always_true(): return True
                    tasks.append(always_true())
            results = await asyncio.gather(*tasks)
            return [stocks[idx].stock_code for idx, needs_update in enumerate(results) if needs_update]
        scheduled_count = 0
        for i in range(0, total_stocks, batch_size):
            batch_stocks = list(all_stocks[i:i+batch_size])
            pending_codes = async_to_sync(_filter_batch_stocks)(batch_stocks)
            for stock_code in pending_codes:
                calculate_fundflow_factors_for_stock.delay(stock_code=stock_code,start_date_str=start_date_str,incremental=incremental)
                scheduled_count += 1
            if pending_codes:
                print(f"已调度第 {i//batch_size + 1} 批，实际下发 {len(pending_codes)} 只增量任务")
            time.sleep(0.05)
        print(f"调度完成: 全市场 {total_stocks} 只股票，实际增量下发 {scheduled_count} 只任务")
        return {'status': 'success', 'message': f'调度成功，实际下发 {scheduled_count} 个任务', 'scheduled_stocks': scheduled_count}
    except Exception as e:
        logger.error(f"调度资金流向因子计算全局异常: {e}", exc_info=True)
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
    版本: V2.1.0
    说明: 性能极限压榨版。消除每次调用时 Django ORM 内部的动态反射开销。
    利用 getattr/setattr 在模型类自身缓存字段集合，将 O(N) 的反射探测损耗彻底降维至 O(1)。
    由于此方法属于标量数据装配，原生 dict 赋值为最优解，禁止引入 Numpy/Numba 以免增加对象创建开销。
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
        model_fields = getattr(factor_model, '_cached_factor_fields', None)
        if model_fields is None:
            model_fields = {f.name for f in factor_model._meta.fields}
            setattr(factor_model, '_cached_factor_fields', model_fields)
        raw_fields = {
            'behavior_pattern', 'divergence_type', 'trading_signal', 'feature_vector', 
            'intraday_flow_distribution', 'large_order_anomaly', 
            'intensity_level', 'inflow_persistence', 'days_since_last_peak',
            'tick_large_order_count', 'flow_persistence_minutes', 'flow_cluster_duration'
        }
        final_data = {}
        for key, value in metrics.items():
            if key in model_fields:
                if key in raw_fields:
                    if key == 'large_order_anomaly' and value is None:
                        final_data[key] = False
                    else:
                        final_data[key] = value
                else:
                    final_data[key] = _safe_decimal(value)
        final_data['stock'] = stock_info
        final_data['trade_time'] = trade_date
        return factor_model(**final_data)
    except Exception as e:
        logger.error(f"构建因子实例失败 {stock_code} @ {trade_date}: {e}")
        return None

def calculate_single_date_factor(stock_code: str, trade_date: date, factor_model, stock_basic_dao, stock_time_trade_dao) -> Optional[Any]:
    """
    版本: V2.1.0
    说明: 同步版性能优化。应用与异步版等同规格的反射缓存策略(Reflection Caching)。
    彻底消除每次遍历生成 instance 时的 _meta.fields 元数据解包耗时。
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
        model_fields = getattr(factor_model, '_cached_factor_fields', None)
        if model_fields is None:
            model_fields = {f.name for f in factor_model._meta.fields}
            setattr(factor_model, '_cached_factor_fields', model_fields)
        raw_fields = {
            'behavior_pattern', 'divergence_type', 'trading_signal', 'feature_vector', 
            'intraday_flow_distribution', 'large_order_anomaly', 
            'intensity_level', 'inflow_persistence', 'days_since_last_peak',
            'tick_large_order_count', 'flow_persistence_minutes', 'flow_cluster_duration'
        }
        final_data = {}
        for key, value in metrics.items():
            if key in model_fields:
                if key in raw_fields:
                    if key == 'large_order_anomaly' and value is None:
                        final_data[key] = False
                    else:
                        final_data[key] = value
                else:
                    final_data[key] = _safe_decimal(value)
        final_data['stock'] = stock_info
        final_data['trade_time'] = trade_date
        return factor_model(**final_data)
    except Exception as e:
        logger.error(f"同步构建因子实例失败 {stock_code} @ {trade_date}: {e}")
        return None

async def _get_historical_flow_data_async(stock_code: str, end_date: date, stock_time_trade_dao, days: int = 120) -> List[Dict]:
    """
    版本: V2.0.0
    说明: 性能重构版。彻底消除 iterrows() 与 Python 层面的 for 循环匹配。
    引入 Pandas 向量化合并 (Vectorized Merge) 与条件计算 (np.where)。
    应用 Downcasting (float64 -> float32) 降低内存带宽占用，提升 CPU 缓存命中率。
    """
    try:
        trade_dates = await sync_to_async(TradeCalendar.get_latest_n_trade_dates)(n=days, reference_date=end_date)
        if not trade_dates:
            return []
        stock_info = await sync_to_async(StockInfo.objects.filter(stock_code=stock_code).first)()
        if not stock_info:
            return []
        sorted_dates = sorted(trade_dates)
        historical_data = []
        for trade_date in sorted_dates:
            flow_data = await sync_to_async(get_single_date_flow_data)(stock_code, trade_date, stock_info)
            if flow_data:
                flow_data['trade_date'] = trade_date.isoformat()
                historical_data.append(flow_data)
        if not historical_data:
            return []
        df_flow = pd.DataFrame(historical_data)
        try:
            start_date = sorted_dates[0]
            real_end_date = sorted_dates[-1]
            s_str = start_date.strftime('%Y%m%d')
            e_str = real_end_date.strftime('%Y%m%d')
            df_price = await stock_time_trade_dao.get_daily_data(stock_code, s_str, e_str)
            if not df_price.empty:
                df_price.index = pd.to_datetime(df_price.index).strftime('%Y-%m-%d')
                df_price = df_price[['close', 'pct_change', 'amount', 'vol']].copy()
                df_price['amount'] = df_price['amount'].fillna(0.0).astype(np.float32)
                df_price['vol'] = df_price['vol'].fillna(0.0).astype(np.float32)
                df_price['close'] = df_price['close'].astype(np.float32)
                df_price['pct_change'] = df_price['pct_change'].astype(np.float32)
                df_flow = df_flow.merge(df_price, left_on='trade_date', right_index=True, how='left')
                df_flow['net_amount_ratio'] = np.where(
                    (df_flow['net_mf_amount'].notnull()) & (df_flow['amount'].fillna(0) > 0),
                    (df_flow['net_mf_amount'] / df_flow['amount']) * 1000.0,
                    0.0
                ).astype(np.float32)
            else:
                df_flow['net_amount_ratio'] = 0.0
                for col in ['close', 'pct_change', 'amount', 'vol']:
                    df_flow[col] = 0.0 if col in ['amount', 'vol'] else None
        except Exception as e:
            logger.error(f"合并股票 {stock_code} 历史行情数据失败: {e}", exc_info=True)
            df_flow['net_amount_ratio'] = 0.0
        return df_flow.replace({np.nan: None}).to_dict('records')
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 历史资金流向数据失败: {e}")
        return []

async def _get_daily_basic_data_async(stock_code: str, trade_date: date, stock_time_trade_dao) -> Optional[Dict]:
    """
    版本: V2.1.0
    说明: 性能优化。重构内部 safe_float 闭包函数，增加 isinstance 黄金短路拦截，
    大幅减少 float() 转换过程中的异常捕获产生的上下文切换开销。
    """
    try:
        daily_data_list = await stock_time_trade_dao.get_stocks_daily_data([stock_code], trade_date)
        if daily_data_list:
            daily_obj = daily_data_list[0]
            def safe_float(val):
                if val is None: return None
                if isinstance(val, (float, int)): return float(val)
                try: return float(val)
                except (ValueError, TypeError): return None
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
    版本: V2.1.0
    说明: 彻底重构增量计算逻辑，引入“区间差集”比对机制。
    废弃原有的“仅依赖最后一条记录向后推算”的逻辑，改为拉取理论区间内的全部交易日，并与数据库中已存在记录作差集。
    无论是否指定 start_date，只要开启 incremental=True，系统都会精准识别并仅计算缺失的日期，彻底解决历史数据空洞问题。
    """
    try:
        latest_dates = TradeCalendar.get_latest_n_trade_dates(n=1, reference_date=timezone.now().date())
        latest_trade_date = latest_dates[0] if latest_dates else None
        if not latest_trade_date:
            logger.warning(f"无法获取股票 {stock_code} 的最新交易日")
            return []
        if start_date:
            calc_start_date = start_date
        else:
            calc_start_date = get_earliest_flow_date(stock_code)
        if not calc_start_date:
            logger.warning(f"无法确定股票 {stock_code} 的计算开始日期")
            return []
        if calc_start_date > latest_trade_date:
            logger.debug(f"股票 {stock_code} 开始日期 {calc_start_date} 晚于最新交易日 {latest_trade_date}")
            return []
        all_trade_dates = TradeCalendar.get_trade_dates_between(start_date=calc_start_date, end_date=latest_trade_date)
        if not all_trade_dates:
            logger.debug(f"股票 {stock_code} 在指定区间内没有理论交易日")
            return []
        if incremental:
            existing_records = factor_model.objects.filter(stock__stock_code=stock_code, trade_time__gte=calc_start_date, trade_time__lte=latest_trade_date).values_list('trade_time', flat=True)
            existing_dates = set(existing_records)
            pending_dates = [d for d in all_trade_dates if d not in existing_dates]
        else:
            pending_dates = all_trade_dates
        logger.debug(f"股票 {stock_code} 理论交易日: {len(all_trade_dates)}天, 实际待计算(增量差集): {len(pending_dates)}天")
        return sorted(pending_dates)
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

def get_historical_flow_data(stock_code: str, end_date: date, stock_time_trade_dao, stock_info, days: int = 120) -> List[Dict]:
    """
    版本: V2.0.0
    说明: 同步获取历史数据的性能重构版。
    使用 pandas 向量化计算替代 iterrows 循环，并对数值列进行 float32 降级，
    大幅削减构建上下文时的计算耗时与内存分配。
    """
    try:
        trade_dates = TradeCalendar.get_latest_n_trade_dates(n=days, reference_date=end_date)
        if not trade_dates or not stock_info:
            return []
        sorted_dates = sorted(trade_dates)
        historical_data = []
        for trade_date in sorted_dates:
            flow_data = get_single_date_flow_data(stock_code, trade_date, stock_info)
            if flow_data:
                flow_data['trade_date'] = trade_date.isoformat()
                historical_data.append(flow_data)
        if not historical_data:
            return []
        df_flow = pd.DataFrame(historical_data)
        try:
            start_date = sorted_dates[0]
            real_end_date = sorted_dates[-1]
            s_str = start_date.strftime('%Y%m%d')
            e_str = real_end_date.strftime('%Y%m%d')
            df_price = async_to_sync(stock_time_trade_dao.get_daily_data)(stock_code, s_str, e_str)
            if not df_price.empty:
                df_price.index = pd.to_datetime(df_price.index).strftime('%Y-%m-%d')
                df_price = df_price[['close', 'pct_change', 'amount', 'vol']].copy()
                df_price['amount'] = df_price['amount'].fillna(0.0).astype(np.float32)
                df_price['vol'] = df_price['vol'].fillna(0.0).astype(np.float32)
                df_price['close'] = df_price['close'].astype(np.float32)
                df_price['pct_change'] = df_price['pct_change'].astype(np.float32)
                df_flow = df_flow.merge(df_price, left_on='trade_date', right_index=True, how='left')
                df_flow['net_amount_ratio'] = np.where(
                    (df_flow['net_mf_amount'].notnull()) & (df_flow['amount'].fillna(0) > 0),
                    (df_flow['net_mf_amount'] / df_flow['amount']) * 1000.0,
                    0.0
                ).astype(np.float32)
            else:
                logger.warning(f"股票 {stock_code} 在 {s_str}-{e_str} 期间无日线行情数据")
                df_flow['net_amount_ratio'] = 0.0
                for col in ['close', 'pct_change', 'amount', 'vol']:
                    df_flow[col] = 0.0 if col in ['amount', 'vol'] else None
        except Exception as e:
            logger.error(f"合并股票 {stock_code} 历史行情数据失败: {e}", exc_info=True)
            df_flow['net_amount_ratio'] = 0.0
        return df_flow.replace({np.nan: None}).to_dict('records')
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

def get_daily_basic_data(stock_code: str, trade_date: date, stock_time_trade_dao: StockTimeTradeDAO) -> Optional[Dict]:
    """
    版本: V2.1.0
    说明: 性能优化。重构内部 safe_float，应用类型短路拦截合法类型，避免异常触发开销。
    保持单行标量提取最高效率，拒绝无意义的向量化强转。
    """
    try:
        daily_data_list = async_to_sync(stock_time_trade_dao.get_stocks_daily_data)([stock_code], trade_date)
        if daily_data_list:
            daily_obj = daily_data_list[0]
            def safe_float(val):
                if val is None: return None
                if isinstance(val, (float, int)): return float(val)
                try: return float(val)
                except (ValueError, TypeError): return None
            basic_dict = {
                'close': safe_float(daily_obj.close),
                'amount': safe_float(daily_obj.amount),
                'vol': safe_float(daily_obj.vol),
                'turnover_rate': safe_float(getattr(daily_obj, 'turnover_rate', None)),
            }
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
    """
    版本: V2.2.0
    说明: 性能极限压榨版。拦截底层返回的1分钟高频数据，执行强制数据类型降级 (Downcasting)。
    将默认的 float64/int64 精度缩减至 float32/int32，直接将内存占用减半。
    此举可极大降低下游计算组件 (Calculator) 在高频矩阵运算时的 CPU 缓存未命中率 (Cache Miss Rate)。
    """
    try:
        df = await stock_time_trade_dao.get_1_min_kline_time_by_day(stock_code, trade_date)
        if df is not None and not df.empty:
            float_cols = df.select_dtypes(include=['float64']).columns
            if len(float_cols) > 0:
                df[float_cols] = df[float_cols].astype(np.float32)
            int_cols = df.select_dtypes(include=['int64']).columns
            if len(int_cols) > 0:
                df[int_cols] = df[int_cols].astype(np.int32)
        return df
    except Exception as e:
        logger.debug(f"获取股票 {stock_code} 在 {trade_date} 的1分钟数据失败: {e}")
        return None

def get_1min_data(stock_code: str, trade_date: date, stock_time_trade_dao: StockTimeTradeDAO) -> Optional[pd.DataFrame]:
    """
    版本: V2.2.0
    说明: 同步获取1分钟数据的性能降维版。
    利用 Pandas 引擎在数据组装源头直接进行 float64 -> float32 与 int64 -> int32 的降级转换，
    以 O(N) 的极小代价换取下游庞大矩阵计算时 O(N^2) 级别的访存加速。
    """
    try:
        df = async_to_sync(stock_time_trade_dao.get_1_min_kline_time_by_day)(stock_code, trade_date)
        if df is not None and not df.empty:
            float_cols = df.select_dtypes(include=['float64']).columns
            if len(float_cols) > 0:
                df[float_cols] = df[float_cols].astype(np.float32)
            int_cols = df.select_dtypes(include=['int64']).columns
            if len(int_cols) > 0:
                df[int_cols] = df[int_cols].astype(np.int32)
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
    """
    版本: V2.1.0
    说明: 性能优化。去除了低效的全面 try-except 捕获。
    通过 isinstance 类型检查构建短路拦截机制 (Short-circuiting)，加速合法数值转换，
    避免 Python 异常堆栈展开在每股每天几十个因子上造成的累积 CPU 阻塞。
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return Decimal(str(value))
    try:
        return Decimal(str(value))
    except (ValueError, TypeError, Exception):
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












