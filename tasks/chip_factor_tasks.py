# tasks\chip_factor_tasks.py
from chaoyue_dreams.celery import app as celery_app
from django.apps import apps
from celery import group, chain, chord
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time as dt_time
import asyncio
from asgiref.sync import sync_to_async, async_to_sync
import logging
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.chip_holding_calculator import AdvancedChipDynamicsService
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
import time

# 导入模型和工具
from stock_models.chip_factors import (
    ChipFactorSZ, ChipFactorSH, ChipFactorCY, ChipFactorKC, ChipFactorBJ, 
)
from stock_models.chip import StockCyqPerf
from stock_models.index import TradeCalendar
from utils.model_helpers import (
    get_cyq_chips_model_by_code,
    get_daily_data_model_by_code,
    get_chip_factor_model_by_code,
    get_chip_holding_matrix_model_by_code,
    get_stock_tick_data_model_by_code,
)
from services.chip_calculator import ChipFactorCalculator
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from utils.cache_manager import CacheManager
from stock_models.stock_basic import StockInfo

logger = logging.getLogger(__name__)

# ========== 配置参数 ==========
class ChipTaskConfig:
    """筹码任务配置"""
    # 队列设置
    QUEUE_NAME = 'calculator'
    # 任务优先级 (0-9, 0最高)
    PRIORITY_HIGH = 0
    PRIORITY_MEDIUM = 5
    PRIORITY_LOW = 9
    # 批量处理大小
    BATCH_SIZE_SINGLE = 100  # 单只股票每日批量大小
    BATCH_SIZE_BULK = 10    # 批量股票每批数量
    # 重试配置
    MAX_RETRIES = 3
    RETRY_DELAY = 60  # 秒
    # 历史数据回溯天数
    HISTORICAL_DAYS_FOR_MA = 120  # 计算MA需要的历史天数
    HISTORICAL_DAYS_FOR_TREND = 20  # 计算趋势需要的历史天数
    # 默认日期范围
    DEFAULT_START_DATE = '20230901'  # 数据从2023年9月1日开始
    @classmethod
    def get_queue_name(cls):
        return cls.QUEUE_NAME

# ========== 辅助函数 ==========
def get_market_from_code(stock_code: str) -> str:
    """从股票代码获取市场标识"""
    if stock_code.endswith('.SZ'):
        if stock_code.startswith('3'):
            return 'CY'  # 创业板
        else:
            return 'SZ'  # 深主板
    elif stock_code.endswith('.SH'):
        if stock_code.startswith('68'):
            return 'KC'  # 科创板
        else:
            return 'SH'  # 沪主板
    elif stock_code.endswith('.BJ'):
        return 'BJ'  # 北交所
    else:
        return 'SZ'  # 默认

def get_last_trade_date() -> date:
    """获取最近一个交易日"""
    # 使用 TradeCalendar 获取最近交易日
    last_trade_date = TradeCalendar.get_latest_trade_date()
    if last_trade_date:
        return last_trade_date
    # 如果获取失败，返回昨天
    return (datetime.now() - timedelta(days=1)).date()

def date_range(start_date: date, end_date: date) -> List[date]:
    """生成交易日范围列表（使用 TradeCalendar）"""
    try:
        # 获取日期范围内的所有交易日
        trade_dates = TradeCalendar.get_trade_dates_between(start_date, end_date)
        print(f"📅 [交易日历] 获取 {start_date} 到 {end_date} 的交易日: {len(trade_dates)} 天")
        if trade_dates:
            print(f"📅 [交易日历] 最早交易日: {trade_dates[0]}, 最晚交易日: {trade_dates[-1]}")
        return trade_dates
    except Exception as e:
        print(f"⚠️ [交易日历] 获取交易日失败: {e}, 使用自然日")
        # 降级方案：使用自然日
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        return dates

def parse_date(date_str: str) -> date:
    """解析日期字符串"""
    try:
        return datetime.strptime(date_str, '%Y%m%d').date()
    except ValueError:
        return datetime.strptime(date_str, '%Y-%m-%d').date()

def get_last_trade_date() -> date:
    """获取最近一个交易日"""
    # 这里需要根据您的交易日历实现
    # 简化：返回昨天
    return TradeCalendar.get_latest_trade_date()

# ========== 调度任务 ==========
@celery_app.task(bind=True, name='tasks.chip_factor_tasks.schedule_chip_factor_calculation', queue=ChipTaskConfig.get_queue_name())
def schedule_chip_factor_calculation(
    self,  
    stock_codes: Optional[List[str]] = None, 
    start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = None, 
    batch_mode: bool = True,
    include_energy_analysis: bool = True,
    calculation_mode: str = 'comprehensive',
    incremental: bool = True
) -> Dict:
    # [V3.1.0] 新增增量调度逻辑，透传incremental参数避免全量遍历浪费IO
    try:
        logger.info(f"开始调度筹码因子计算任务（模式: {calculation_mode}, 增量模式: {incremental}）")
        if start_date_str:
            start_date = parse_date(start_date_str)
        else:
            start_date = parse_date(ChipTaskConfig.DEFAULT_START_DATE)
        if end_date_str:
            end_date = parse_date(end_date_str)
        else:
            end_date = get_last_trade_date()
        logger.info(f"全局请求日期范围: {start_date} 到 {end_date}")
        if stock_codes is None:
            cache_manager = CacheManager()
            stock_dao = StockBasicInfoDao(cache_manager)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                stock_list = loop.run_until_complete(stock_dao.get_stock_list())
                stock_codes = [stock.stock_code for stock in stock_list]
            finally:
                loop.close()
        logger.info(f"需要计算的股票数量: {len(stock_codes)}")
        if calculation_mode == 'comprehensive':
            result = schedule_comprehensive_calculation(
                stock_codes, start_date, end_date, batch_mode, incremental
            )
        elif calculation_mode == 'chip_only':
            if batch_mode:
                result = schedule_by_stock_batch(stock_codes, start_date, end_date, incremental)
            else:
                result = schedule_by_date_batch(stock_codes, start_date, end_date)
        elif calculation_mode == 'energy_only':
            result = schedule_energy_only_calculation(stock_codes, start_date, end_date, incremental)
        else:
            raise ValueError(f"未知的计算模式: {calculation_mode}")
        return result
    except Exception as e:
        logger.error(f"调度筹码因子计算任务失败: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=ChipTaskConfig.RETRY_DELAY)

def schedule_by_stock_batch(stock_codes: List[str], start_date: date, end_date: date, incremental: bool = True) -> Dict:
    # [V3.1.0] 按股票调度注入incremental参数
    total_tasks = 0
    market_groups = {}
    for code in stock_codes:
        market = get_market_from_code(code)
        market_groups.setdefault(market, []).append(code)
    for market, codes in market_groups.items():
        for i in range(0, len(codes), ChipTaskConfig.BATCH_SIZE_BULK):
            batch_codes = codes[i:i + ChipTaskConfig.BATCH_SIZE_BULK]
            task = calculate_chip_factors_batch.delay(
                stock_codes=batch_codes, 
                start_date=start_date.strftime('%Y%m%d'), 
                end_date=end_date.strftime('%Y%m%d'), 
                market=market,
                incremental=incremental
            )
            total_tasks += 1
    return {'status': 'scheduled', 'total_tasks': total_tasks, 'mode': 'stock_batch', 'total_stocks': len(stock_codes)}

def schedule_by_date_batch(stock_codes: List[str], start_date: date, end_date: date) -> Dict:
    """按日期分批调度"""
    total_tasks = 0
    # 生成所有交易日（这里简化，实际需要交易日历）
    all_dates = date_range(start_date, end_date)
    # 按日期分批
    for current_date in all_dates:
        # 为每个日期创建任务
        task = calculate_chip_factors_for_date.delay(
            trade_date_str=current_date.strftime('%Y%m%d'),
            stock_codes=stock_codes
        )
        total_tasks += 1
        logger.debug(f"创建日期计算任务 {task.id}: {current_date}")
    return {
        'status': 'scheduled',
        'total_tasks': total_tasks,
        'mode': 'date_batch',
        'date_range': f"{start_date} - {end_date}"
    }

# ========== 批量计算任务 ==========
@celery_app.task(bind=True, name='tasks.chip_factor_tasks.calculate_chip_factors_batch', queue=ChipTaskConfig.get_queue_name())
def calculate_chip_factors_batch(self, stock_codes: List[str], start_date: str, end_date: str, market: str = None, incremental: bool = True) -> Dict:
    # [V3.1.0] 引入O(1)增量探底逻辑，精准定位每只股票的断点，避免无效的日期历遍查询
    try:
        logger.info(f"开始批量计算筹码因子，股票数量: {len(stock_codes)}，增量模式: {incremental}")
        global_start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        results = {'total': len(stock_codes), 'success': 0, 'failed': 0, 'details': []}
        print(f"📋 [批量任务] 开始处理 {len(stock_codes)} 只股票")
        for stock_index, stock_code in enumerate(stock_codes):
            try:
                print(f"🔴 [股票处理] 开始处理第 {stock_index + 1}/{len(stock_codes)} 只股票: {stock_code}")
                actual_start_date_obj = global_start_date_obj
                if incremental:
                    chip_factor_model = get_chip_factor_model_by_code(stock_code)
                    latest_record = chip_factor_model.objects.filter(
                        stock__stock_code=stock_code,
                        calc_status='success'
                    ).order_by('-trade_time').first()
                    if latest_record:
                        next_trade_date = TradeCalendar.get_next_trade_date(latest_record.trade_time)
                        if next_trade_date:
                            actual_start_date_obj = max(global_start_date_obj, next_trade_date)
                            print(f"🔍 [增量探底] {stock_code} 最新成功日期: {latest_record.trade_time}, 调整起始日为: {actual_start_date_obj}")
                if actual_start_date_obj > end_date_obj:
                    print(f"✅ [增量跳过] {stock_code} 已更新至最新日期 {latest_record.trade_time}，无需计算")
                    results['success'] += 1
                    results['details'].append({'stock_code': stock_code, 'stock_index': stock_index + 1, 'status': 'success', 'processed_dates': 0, 'message': '已是最新'})
                    continue
                HoldingMatrixModel = get_chip_holding_matrix_model_by_code(stock_code)
                date_range_count = HoldingMatrixModel.objects.filter(
                    stock__stock_code=stock_code,
                    trade_time__gte=actual_start_date_obj,
                    trade_time__lte=end_date_obj,
                    calc_status='success'
                ).count()
                if date_range_count == 0:
                    print(f"⚠️ [股票检查] {stock_code} 在日期范围内无成功计算的持有矩阵，筹码因子将使用默认值")
                result = calculate_single_stock_chip_factors_sync(stock_code, actual_start_date_obj, end_date_obj)
                if result.get('status') == 'success':
                    results['success'] += 1
                    processed_dates = result.get('processed_dates', 0)
                    print(f"✅ [股票完成] {stock_code} 处理完成，成功 {processed_dates} 个交易日")
                else:
                    results['failed'] += 1
                    print(f"❌ [股票失败] {stock_code} 处理失败: {result.get('error', '未知错误')}")
                results['details'].append({'stock_code': stock_code, 'stock_index': stock_index + 1, **result})
            except Exception as e:
                results['failed'] += 1
                logger.error(f"股票 {stock_code} 计算异常: {e}")
                results['details'].append({'stock_code': stock_code, 'stock_index': stock_index + 1, 'status': 'error', 'error': str(e), 'processed_dates': 0})
        return results
    except Exception as e:
        logger.error(f"批量计算筹码因子失败: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=ChipTaskConfig.RETRY_DELAY)

def calculate_single_stock_chip_factors_sync(stock_code: str, start_date: date, end_date: date) -> Dict:
    """同步版本的单个股票计算函数（按股票循环）"""
    try:
        print(f"🔴 [单股开始] 开始处理股票 {stock_code}")
        print(f"📅 [单股日期] 日期范围: {start_date} 到 {end_date}")
        # 创建事件循环用于异步调用
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            start_time = time.time()
            result = loop.run_until_complete(calculate_single_stock_chip_factors_async(stock_code, start_date, end_date))
            elapsed_time = time.time() - start_time
            return result
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"同步计算股票 {stock_code} 失败: {e}")
        print(f"❌ [单股异常] {stock_code}: {e}")
        return {'status': 'error', 'error': str(e), 'processed_dates': 0}

async def calculate_single_stock_chip_factors_async(stock_code: str, start_date: date, end_date: date) -> Dict:
    # [V3.1.2] 优化说明：引入from_records和float32数据类型强制降级；优化小范围数值统计性能
    try:
        logger.debug(f"开始计算股票 {stock_code} 的筹码因子")
        print(f"🔴 [单股异步开始] {stock_code} {start_date} 到 {end_date}")
        cm = CacheManager()
        realtime_dao = StockRealtimeDAO(cm)
        HoldingMatrixModel = get_chip_holding_matrix_model_by_code(stock_code)
        holding_count = await sync_to_async(HoldingMatrixModel.objects.filter(stock__stock_code=stock_code,trade_time__gte=start_date,trade_time__lte=end_date,calc_status='success').count)()
        if holding_count == 0:
            print(f"⚠️ [单股检查] {stock_code} 在日期范围内无持有矩阵数据，将尝试降级计算")
        else:
            print(f"✅ [单股检查] {stock_code} 已有 {holding_count} 天的持有矩阵数据")
        chip_factor_model = get_chip_factor_model_by_code(stock_code)
        chips_model = get_cyq_chips_model_by_code(stock_code)
        daily_data_model = get_daily_data_model_by_code(stock_code)
        stock = await sync_to_async(StockInfo.objects.filter(stock_code=stock_code).first)()
        if not stock:
            print(f"❌ [单股错误] {stock_code} 股票不存在")
            return {'status': 'failed', 'error': f'未找到股票 {stock_code}', 'processed_dates': 0}
        historical_df = await get_historical_prices_for_stock(stock_code, end_date, ChipTaskConfig.HISTORICAL_DAYS_FOR_MA)
        if historical_df.empty:
            print(f"❌ [单股错误] {stock_code} 历史价格数据不足")
            return {'status': 'failed', 'error': f'股票 {stock_code} 历史价格数据不足', 'processed_dates': 0}
        print(f"📊 [单股数据] 历史价格: {len(historical_df)} 条")
        historical_prices_series = historical_df['close_qfq'] if 'close_qfq' in historical_df else pd.Series(dtype=np.float32)
        get_dates_between_func = sync_to_async(TradeCalendar.get_trade_dates_between, thread_sensitive=True)
        trade_dates = await get_dates_between_func(start_date, end_date)
        if not trade_dates:
            print(f"⚠️ [单股警告] {stock_code} 日期范围内无交易日: {start_date} 到 {end_date}")
            return {'status': 'failed', 'error': '日期范围内无交易日', 'processed_dates': 0}
        processed_dates = 0
        saved_dates = []
        failed_dates = []
        date_progress_interval = max(1, len(trade_dates) // 10)
        for date_index, current_date in enumerate(trade_dates):
            try:
                if date_index % date_progress_interval == 0:
                    progress = (date_index + 1) / len(trade_dates) * 100
                    print(f"📊 [单股进度] {stock_code} 进度: {progress:.1f}% ({date_index + 1}/{len(trade_dates)})")
                existing = await sync_to_async(chip_factor_model.objects.filter(stock=stock, trade_time=current_date, calc_status='success').exists)()
                if existing:
                    continue
                chip_perf = await sync_to_async(StockCyqPerf.objects.filter(stock=stock, trade_time=current_date).first)()
                if not chip_perf:
                    continue
                chips_data = await sync_to_async(list)(chips_model.objects.filter(stock=stock, trade_time=current_date).values('price', 'percent'))
                if not chips_data:
                    continue
                chips_df = pd.DataFrame.from_records(chips_data).astype(np.float32)
                daily_kline = await sync_to_async(daily_data_model.objects.filter(stock=stock, trade_time=current_date).first)()
                if not daily_kline:
                    continue
                get_offset_func = sync_to_async(TradeCalendar.get_trade_date_offset, thread_sensitive=True)
                prev_date = await get_offset_func(current_date, -1)
                if prev_date:
                    prev_chips_data = await sync_to_async(list)(chips_model.objects.filter(stock=stock, trade_time=prev_date).values('price', 'percent'))
                    prev_chips_df = pd.DataFrame.from_records(prev_chips_data).astype(np.float32) if prev_chips_data else pd.DataFrame()
                else:
                    prev_chips_df = pd.DataFrame()
                historical_factors = await get_historical_chip_factors(chip_factor_model, stock, current_date, 5)
                current_turnover = historical_df.loc[current_date, 'turnover_rate'] if current_date in historical_df.index else 0.0
                tick_data = None
                try:
                    tick_data = await realtime_dao.get_daily_real_ticks(stock_code, current_date)
                except Exception as tick_error:
                    logger.warning(f"获取 {stock_code} {current_date} tick数据失败: {tick_error}")
                chip_perf_dict = {'weight_avg': chip_perf.weight_avg, 'his_high': chip_perf.his_high, 'his_low': chip_perf.his_low, 'cost_5pct': chip_perf.cost_5pct, 'cost_15pct': chip_perf.cost_15pct, 'cost_50pct': chip_perf.cost_50pct, 'cost_85pct': chip_perf.cost_85pct, 'cost_95pct': chip_perf.cost_95pct, 'winner_rate': chip_perf.winner_rate}
                daily_kline_dict = {'close': daily_kline.close_qfq, 'open': daily_kline.open_qfq, 'high': daily_kline.high_qfq, 'low': daily_kline.low_qfq, 'vol': daily_kline.vol, 'amount': daily_kline.amount, 'pct_change': daily_kline.pct_change}
                daily_basic_dict = {'turnover_rate': current_turnover}
                try:
                    factors = ChipFactorCalculator.calculate_complete_factors_with_tick(chip_perf_data=chip_perf_dict, chip_dist_data=chips_df, daily_basic_data=daily_basic_dict, daily_kline_data=daily_kline_dict, prev_chip_dist_data=prev_chips_df, historical_prices=historical_prices_series, historical_chip_factors=historical_factors,tick_data=tick_data)
                except Exception as calc_error:
                    logger.error(f"使用tick数据计算因子失败，回退: {calc_error}")
                    factors = ChipFactorCalculator.calculate_complete_factors(chip_perf_data=chip_perf_dict, chip_dist_data=chips_df, daily_basic_data=daily_basic_dict, daily_kline_data=daily_kline_dict, prev_chip_dist_data=prev_chips_df, historical_prices=historical_prices_series, historical_chip_factors=historical_factors)
                try:
                    holding_matrix = await sync_to_async(HoldingMatrixModel.objects.filter(stock=stock,trade_time=current_date,calc_status='success').first)()
                    if holding_matrix:
                        factors['accumulation_signal_score'] = holding_matrix.behavior_accumulation
                        factors['distribution_signal_score'] = holding_matrix.behavior_distribution
                        factors['percent_change_convergence'] = holding_matrix.convergence_comprehensive
                        if holding_matrix.convergence_comprehensive is not None:
                            factors['percent_change_divergence'] = 1.0 - holding_matrix.convergence_comprehensive
                        factors['migration_convergence_ratio'] = holding_matrix.convergence_migration
                        if holding_matrix.extra_metrics:
                            migration = holding_matrix.extra_metrics.get('migration', {})
                            factors['net_migration_direction'] = migration.get('net_migration_direction')
                            behavior_meta = holding_matrix.extra_metrics.get('behavior_meta', {})
                            factors['main_force_activity_index'] = behavior_meta.get('main_force_activity')
                            pressure = holding_matrix.extra_metrics.get('pressure', {})
                            factors['pressure_release_index'] = pressure.get('pressure_release')
                        if holding_matrix.chart_signals:
                            abs_signals = holding_matrix.chart_signals.get('absolute_signals', {})
                            factors['signal_quality_score'] = abs_signals.get('signal_quality')
                            increase_areas = abs_signals.get('significant_increase_areas', [])
                            decrease_areas = abs_signals.get('significant_decrease_areas', [])
                            total_increase = sum(abs(area.get('change', 0)) for area in increase_areas)
                            total_decrease = sum(abs(area.get('change', 0)) for area in decrease_areas)
                            factors['absolute_change_strength'] = (total_increase + total_decrease) / 100.0
                        res_strength = holding_matrix.resistance_strength
                        sup_strength = holding_matrix.support_strength
                        if res_strength and res_strength > 0:
                            factors['support_resistance_ratio'] = (sup_strength or 0) / res_strength
                        else:
                            factors['support_resistance_ratio'] = 1.0 if (sup_strength or 0) > 0 else 0.0
                        conf_score = 0.0
                        conf_count = 0
                        if factors.get('accumulation_signal_score', 0) > 0:
                            conf_score += factors['accumulation_signal_score']
                            conf_count += 1
                        if factors.get('distribution_signal_score', 0) > 0:
                            conf_score += factors['distribution_signal_score']
                            conf_count += 1
                        if factors.get('net_migration_direction') and abs(factors['net_migration_direction']) > 0.1:
                            conf_score += min(1.0, abs(factors['net_migration_direction']) / 10.0)
                            conf_count += 1
                        factors['behavior_confirmation'] = conf_score / conf_count if conf_count > 0 else 0.0
                        if holding_matrix.intraday_market_microstructure:
                            micro = holding_matrix.intraday_market_microstructure
                            tick_fields = ['intraday_chip_consolidation_degree','intraday_peak_valley_ratio','intraday_price_distribution_skewness','intraday_resistance_test_count','intraday_support_test_count','intraday_trough_filling_degree','tick_abnormal_volume_ratio','tick_chip_balance_ratio','tick_chip_transfer_efficiency','tick_clustering_index','tick_level_chip_flow','intraday_chip_concentration','intraday_chip_entropy','intraday_chip_turnover_intensity','intraday_cost_center_migration','intraday_cost_center_volatility','intraday_low_lock_ratio','intraday_high_lock_ratio','intraday_chip_game_index']
                            for field in tick_fields:
                                if field in micro: factors[field] = micro[field]
                except Exception as sync_error:
                    logger.warning(f"从持有矩阵同步因子失败 {stock_code} {current_date}: {sync_error}")
                await save_chip_factors(chip_factor_model, stock, current_date, factors)
                verify_result = await verify_chip_factor_saved(stock_code, current_date)
                if verify_result.get('exists'):
                    processed_dates += 1
                    saved_dates.append(current_date)
                else:
                    failed_dates.append(current_date)
            except Exception as e:
                logger.warning(f"股票 {stock_code} 日期 {current_date} 计算失败: {e}")
                failed_dates.append(current_date)
        print(f"📊 [单股统计] 成功: {len(saved_dates)} 天, 失败: {len(failed_dates)} 天")
        return {'status': 'success', 'processed_dates': processed_dates, 'saved_dates': len(saved_dates), 'failed_dates': len(failed_dates), 'date_range': f"{start_date} - {end_date}", 'tick_stats': {'total_dates': len(trade_dates), 'tick_available_dates': 0, 'daily_approximation_dates': 0}}
    except Exception as e:
        logger.error(f"计算股票 {stock_code} 筹码因子失败: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e), 'processed_dates': 0}

async def get_historical_prices_for_stock(stock_code: str, end_date: date, days: int) -> pd.DataFrame:
    # [V3.1.2] 优化说明：合并重复的获取历史价格方法；使用from_records加速DF构建；实施float32数据类型降级，内存占用减半
    try:
        daily_data_model = get_daily_data_model_by_code(stock_code)
        start_date = end_date - timedelta(days=days * 2)
        price_data = await sync_to_async(list)(daily_data_model.objects.filter(stock__stock_code=stock_code,trade_time__gte=start_date,trade_time__lte=end_date).order_by('trade_time').values('trade_time', 'close_qfq'))
        if not price_data:
            return pd.DataFrame()
        df_price = pd.DataFrame.from_records(price_data)
        df_price['trade_time'] = pd.to_datetime(df_price['trade_time']).dt.date
        df_price.set_index('trade_time', inplace=True)
        df_price['close_qfq'] = df_price['close_qfq'].astype(np.float32)
        try:
            StockDailyBasic = apps.get_model('stock_models', 'StockDailyBasic')
            basic_data = await sync_to_async(list)(StockDailyBasic.objects.filter(stock__stock_code=stock_code,trade_time__gte=start_date,trade_time__lte=end_date).order_by('trade_time').values('trade_time', 'turnover_rate_f', 'volume_ratio'))
            if basic_data:
                df_basic = pd.DataFrame.from_records(basic_data)
                df_basic['trade_time'] = pd.to_datetime(df_basic['trade_time']).dt.date
                df_basic.set_index('trade_time', inplace=True)
                df_basic.rename(columns={'turnover_rate_f': 'turnover_rate'}, inplace=True)
                df_merged = df_price.join(df_basic, how='left')
                if 'turnover_rate' in df_merged.columns:
                    df_merged['turnover_rate'] = df_merged['turnover_rate'].astype(float).fillna(0.0).astype(np.float32)
                else:
                    df_merged['turnover_rate'] = np.float32(0.0)
                if 'volume_ratio' in df_merged.columns:
                    df_merged['volume_ratio'] = df_merged['volume_ratio'].astype(float).astype(np.float32)
                else:
                    df_merged['volume_ratio'] = np.float32(np.nan)
                return df_merged
            else:
                df_price['turnover_rate'] = np.float32(0.0)
                df_price['volume_ratio'] = np.float32(np.nan)
                return df_price
        except LookupError:
            logger.error("无法找到 StockDailyBasic 模型")
            df_price['turnover_rate'] = np.float32(0.0)
            df_price['volume_ratio'] = np.float32(np.nan)
            return df_price
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 历史数据失败: {e}")
        return pd.DataFrame()

def calculate_single_stock_holding_matrix_sync(stock_code: str, start_date: date, end_date: date) -> Dict:
    """同步版本的单个股票持有矩阵计算函数（按股票循环）版本：重构适配AdvancedChipDynamicsService"""
    try:
        print(f"🔴 [持有矩阵单股开始] 开始处理股票 {stock_code}")
        print(f"📅 [持有矩阵单股日期] 日期范围: {start_date} 到 {end_date}")
        # 创建事件循环用于异步调用
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cm = CacheManager()
        try:
            result = loop.run_until_complete(calculate_single_stock_holding_matrix_async(stock_code, start_date, end_date, cm))
            return result
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"同步计算股票 {stock_code} 持有矩阵失败: {e}")
        print(f"❌ [持有矩阵单股异常] {stock_code}: {e}")
        return {'status': 'error', 'error': str(e), 'processed_dates': 0}

async def calculate_single_stock_holding_matrix_async(stock_code: str, start_date: date, end_date: date, cm: CacheManager) -> Dict:
    """异步版本的单个股票持有矩阵计算函数 - 修复数据保存问题"""
    try:
        logger.info(f"开始计算股票 {stock_code} 的持有时间矩阵")
        # 获取持有矩阵模型
        holding_matrix_model = get_chip_holding_matrix_model_by_code(stock_code)
        # 获取股票基本信息
        stock = await sync_to_async(StockInfo.objects.filter(stock_code=stock_code).first)()
        if not stock:
            print(f"❌ [持有矩阵] {stock_code} 股票不存在")
            return {'status': 'failed', 'error': f'未找到股票 {stock_code}', 'processed_dates': 0}
        # 创建动态分析服务
        service = AdvancedChipDynamicsService(market_type=get_market_from_code(stock_code))
        processed_dates = 0
        saved_dates = []
        failed_dates = []
        # 获取日期范围内的所有交易日
        get_dates_func = sync_to_async(TradeCalendar.get_trade_dates_between, thread_sensitive=True)
        trade_dates = await get_dates_func(start_date, end_date)
        if not trade_dates:
            print(f"⚠️ [持有矩阵] {stock_code} 日期范围内无交易日: {start_date} 到 {end_date}")
            return {'status': 'failed', 'error': '日期范围内无交易日', 'processed_dates': 0}
        print(f"📅 [持有矩阵] {stock_code} 交易日: {len(trade_dates)} 天")
        # 按日期循环处理当前股票
        for date_index, current_date in enumerate(trade_dates):
            try:
                if (date_index + 1) % max(1, len(trade_dates) // 10) == 0:
                    progress = (date_index + 1) / len(trade_dates) * 100
                # 检查是否已计算（只检查成功状态）
                existing = await sync_to_async(holding_matrix_model.objects.filter(
                    stock=stock, trade_time=current_date, calc_status='success'
                ).exists)()
                if existing:
                    # 即使已存在，也检查关键字段是否为空
                    existing_record = await sync_to_async(holding_matrix_model.objects.filter(
                        stock=stock, trade_time=current_date
                    ).first)()
                    # 如果关键字段为空，重新计算
                    need_recalculate = False
                    if existing_record:
                        if (not hasattr(existing_record, 'absolute_change_analysis') or 
                            existing_record.absolute_change_analysis is None or
                            existing_record.absolute_change_analysis == {}):
                            need_recalculate = True
                            print(f"🔄 [持有矩阵] {stock_code} {current_date} absolute_change_analysis为空，重新计算")
                        elif (not hasattr(existing_record, 'absorption_energy') or 
                              existing_record.absorption_energy is None or
                              existing_record.absorption_energy == 0):
                            need_recalculate = True
                            print(f"🔄 [持有矩阵] {stock_code} {current_date} 能量场数据为空，重新计算")
                    if not need_recalculate:
                        continue
                # 使用AdvancedChipDynamicsService进行动态分析
                trade_date_str = current_date.strftime('%Y-%m-%d')
                realtime_dao = StockRealtimeDAO(cm)
                tick_data = await realtime_dao.get_daily_real_ticks(stock_code, current_date) 
                dynamics_result = await service.analyze_chip_dynamics_daily(
                    stock_code=stock_code,
                    trade_date=trade_date_str,
                    lookback_days=20,
                    tick_data=tick_data
                )
                # 保存动态分析结果到数据库
                if dynamics_result.get('analysis_status') == 'success':
                    # 获取或创建记录
                    record, created = await sync_to_async(holding_matrix_model.objects.get_or_create)(
                        stock=stock,
                        trade_time=current_date,
                        defaults={'calc_status': 'pending'}
                    )
                    # 确保dynamics_result包含current_price
                    if 'current_price' not in dynamics_result:
                        # 尝试从其他字段推断
                        if 'price_grid' in dynamics_result and dynamics_result['price_grid']:
                            price_grid = dynamics_result['price_grid']
                            if len(price_grid) > 0:
                                # 使用价格网格的中间值作为当前价
                                dynamics_result['current_price'] = price_grid[len(price_grid)//2]
                    save_success = await sync_to_async(record.save_dynamics_result)(dynamics_result)
                    if save_success:
                        processed_dates += 1
                        saved_dates.append(current_date)
                    else:
                        failed_dates.append(current_date)
                        print(f"❌ [持有矩阵] {stock_code} {current_date} 动态分析保存失败")
                else:
                    failed_dates.append(current_date)
                    print(f"⚠️ [持有矩阵] {stock_code} {current_date} 动态分析失败: {dynamics_result.get('analysis_status', 'unknown')}")
                    
            except Exception as e:
                print(f"❌ [持有矩阵] {stock_code} {current_date} 计算失败: {e}")
                import traceback
                traceback.print_exc()
                failed_dates.append(current_date)
        print(f"✅ [持有矩阵完成] {stock_code} 处理完成")
        print(f"📊 [持有矩阵统计] 成功: {len(saved_dates)} 个交易日，失败: {len(failed_dates)} 个交易日")
        return {
            'status': 'success', 
            'processed_dates': processed_dates, 
            'saved_dates': len(saved_dates), 
            'failed_dates': len(failed_dates), 
            'date_range': f"{start_date} - {end_date}"
        }
    except Exception as e:
        logger.error(f"计算股票 {stock_code} 持有矩阵失败: {e}", exc_info=True)
        print(f"❌ [持有矩阵异常] {stock_code}: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e), 'processed_dates': 0}

def calculate_holding_matrix_for_stock_sync(stock_code: str, start_date: date, end_date: date) -> Dict:
    """同步版本的股票持有矩阵计算（按股票循环）版本：重构适配AdvancedChipDynamicsService"""
    try:
        logger.info(f"开始同步计算股票 {stock_code} 的持有时间矩阵（使用AdvancedChipDynamicsService）")
        # 创建事件循环用于异步调用
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            cm = CacheManager()
            result = loop.run_until_complete(calculate_single_stock_holding_matrix_async(stock_code, start_date, end_date, cm))
            return result
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"同步计算股票 {stock_code} 持有矩阵失败: {e}")
        print(f"❌ [持有矩阵异常] {stock_code}: {e}")
        return {'status': 'error', 'error': str(e), 'processed_dates': 0}

async def get_historical_chip_factors(chip_factor_model,stock,current_date: date,days: int) -> List[Dict]:
    """获取历史筹码因子（包含更多字段）"""
    try:
        start_date = current_date - timedelta(days=days)
        historical_factors = await sync_to_async(list)(
            chip_factor_model.objects.filter(
                stock=stock,
                trade_time__gte=start_date,
                trade_time__lt=current_date,
                calc_status='success'
            ).order_by('trade_time')
            .values('chip_mean', 'chip_stability', 'chip_concentration_ratio')
        )
        return historical_factors
    except Exception as e:
        logger.error(f"获取历史筹码因子失败: {e}")
        return []

async def save_chip_factors(chip_factor_model, stock, trade_date, factors):
    """保存筹码因子到数据库 - 支持tick相关字段"""
    try:
        # 创建或更新记录
        obj, created = await sync_to_async(chip_factor_model.objects.update_or_create)(
            stock=stock,
            trade_time=trade_date,
            defaults=factors
        )
        return obj
    except Exception as e:
        logger.error(f"保存筹码因子失败 {stock.stock_code} {trade_date}: {e}")
        print(f"❌ [保存因子] {stock.stock_code} {trade_date}: 失败 - {e}")
        raise

async def verify_chip_factor_saved(stock_code: str, trade_date: date) -> Dict:
    """验证筹码因子是否已保存到数据库"""
    try:
        # 获取股票
        stock = await sync_to_async(StockInfo.objects.filter(stock_code=stock_code).first)()
        if not stock:
            return {'status': 'failed', 'error': '股票不存在'}
        # 获取对应的因子模型
        chip_factor_model = get_chip_factor_model_by_code(stock_code)
        # 检查记录是否存在
        exists = await sync_to_async(chip_factor_model.objects.filter(stock=stock, trade_time=trade_date).exists)()
        if exists:
            # 获取记录详情
            record = await sync_to_async(chip_factor_model.objects.filter(stock=stock, trade_time=trade_date).first)()
            field_count = len([f for f in chip_factor_model._meta.get_fields() if not f.is_relation])
            result = {
                'status': 'success',
                'exists': True,
                'record_id': record.id if record else None,
                'calc_status': getattr(record, 'calc_status', 'unknown') if record else 'unknown',
                'field_count': field_count,
                'message': f"找到 {stock_code} {trade_date} 的记录"
            }
            # 打印一些关键字段
            if record:
                key_fields = ['chip_mean', 'chip_std', 'chip_concentration_ratio', 
                             'avg_holding_days', 'short_term_chip_ratio', 'long_term_chip_ratio',
                             'main_cost_range_ratio', 'high_position_lock_ratio_90',  # 添加这两个字段
                             'calc_status', 'create_time', 'update_time']
                for field in key_fields:
                    if hasattr(record, field):
                        value = getattr(record, field)
                        result[f'field_{field}'] = value
        else:
            # 检查数据库中是否有任何该股票的记录
            total_count = await sync_to_async(chip_factor_model.objects.filter(stock=stock).count)()
            recent_records = await sync_to_async(list)(
                chip_factor_model.objects.filter(stock=stock)
                .order_by('-trade_time')
                .values('trade_time', 'calc_status')[:5]
            )
            result = {
                'status': 'failed',
                'exists': False,
                'total_records': total_count,
                'recent_records': recent_records,
                'message': f"未找到 {stock_code} {trade_date} 的记录"
            }
            print(f"❌ [验证] 未找到 {stock_code} {trade_date} 的记录")
            print(f"📊 [验证] 该股票共有 {total_count} 条记录")
            if recent_records:
                print(f"📊 [验证] 最近5条记录:")
                for rec in recent_records:
                    print(f"    {rec['trade_time']}: {rec['calc_status']}")
                    
        return result
    except Exception as e:
        print(f"❌ [验证异常] {stock_code} {trade_date}: {e}")
        return {'status': 'error', 'error': str(e)}

# ========== 单日计算任务 ==========
@celery_app.task(bind=True, name='tasks.chip_factor_tasks.calculate_chip_factors_for_date', queue=ChipTaskConfig.get_queue_name())
def calculate_chip_factors_for_date(self, trade_date_str: str, stock_codes: Optional[List[str]] = None) -> Dict:
    """
    计算指定日期的筹码因子
    Args:
        trade_date_str: 交易日期 (YYYYMMDD)
        stock_codes: 股票代码列表，None表示全市场
    Returns:
        Dict: 计算结果
    """
    try:
        trade_date = parse_date(trade_date_str)
        logger.info(f"开始计算 {trade_date} 的筹码因子")
        # 获取需要计算的股票
        if stock_codes is None:
            # 获取全市场股票
            cache_manager = CacheManager()
            stock_dao = StockBasicInfoDao(cache_manager)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                stock_list = loop.run_until_complete(stock_dao.get_stock_list())
                stock_codes = [stock.stock_code for stock in stock_list]
            finally:
                loop.close()
        results = {
            'date': trade_date_str,
            'total': len(stock_codes),
            'success': 0,
            'failed': 0,
            'details': []
        }
        # 分批处理
        batch_size = ChipTaskConfig.BATCH_SIZE_BULK
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i + batch_size]
            # 创建子任务
            task = calculate_date_chip_factors_batch.delay(
                trade_date_str=trade_date_str,
                stock_codes=batch_codes
            )
            # 等待任务完成（可改为异步）
            try:
                task_result = task.get(timeout=300)  # 5分钟超时
                results['success'] += task_result.get('success', 0)
                results['failed'] += task_result.get('failed', 0)
                results['details'].extend(task_result.get('details', []))
            except Exception as e:
                logger.error(f"批次 {i//batch_size + 1} 计算失败: {e}")
                results['failed'] += len(batch_codes)
        logger.info(f"日期 {trade_date} 筹码因子计算完成: {results}")
        return results
    except Exception as e:
        logger.error(f"计算日期 {trade_date_str} 筹码因子失败: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=ChipTaskConfig.RETRY_DELAY)

@celery_app.task(bind=True, name='tasks.chip_factor_tasks.calculate_date_chip_factors_batch', queue=ChipTaskConfig.get_queue_name())
def calculate_date_chip_factors_batch(self, trade_date_str: str, stock_codes: List[str]) -> Dict:
    """计算指定日期批量的筹码因子"""
    try:
        trade_date = parse_date(trade_date_str)
        results = {
            'total': len(stock_codes),
            'success': 0,
            'failed': 0,
            'details': []
        }
        # 使用线程池并行计算
        with ThreadPoolExecutor(max_workers=min(10, len(stock_codes))) as executor:
            future_to_stock = {}
            for stock_code in stock_codes:
                future = executor.submit(
                    calculate_single_stock_single_date_sync,
                    stock_code,
                    trade_date
                )
                future_to_stock[future] = stock_code
            for future in as_completed(future_to_stock):
                stock_code = future_to_stock[future]
                try:
                    result = future.result()
                    if result.get('status') == 'success':
                        results['success'] += 1
                    else:
                        results['failed'] += 1
                    results['details'].append({
                        'stock_code': stock_code,
                        **result
                    })
                except Exception as e:
                    results['failed'] += 1
                    logger.error(f"股票 {stock_code} 日期 {trade_date} 计算异常: {e}")
                    results['details'].append({
                        'stock_code': stock_code,
                        'status': 'error',
                        'error': str(e)
                    })
        return results
    except Exception as e:
        logger.error(f"计算日期批量筹码因子失败: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=ChipTaskConfig.RETRY_DELAY)

def calculate_single_stock_single_date_sync(stock_code: str,trade_date: date) -> Dict:
    """计算单只股票单日筹码因子（同步版本）"""
    try:
        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                calculate_single_stock_single_date_async(stock_code, trade_date)
            )
        finally:
            loop.close()
        return result
    except Exception as e:
        logger.error(f"计算股票 {stock_code} 日期 {trade_date} 失败: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

async def calculate_single_stock_single_date_async(stock_code: str, trade_date: date) -> Dict:
    # [V3.1.2] 优化说明：单日计算接口同步实施from_records与float32数据类型降级策略
    try:
        print(f"🔴 [单日计算] 开始计算 {stock_code} {trade_date}")
        chip_factor_model = get_chip_factor_model_by_code(stock_code)
        chips_model = get_cyq_chips_model_by_code(stock_code)
        daily_data_model = get_daily_data_model_by_code(stock_code)
        stock = await sync_to_async(StockInfo.objects.filter(stock_code=stock_code).first)()
        if not stock: return {'status': 'failed', 'error': '股票不存在'}
        existing = await sync_to_async(chip_factor_model.objects.filter(stock=stock, trade_time=trade_date, calc_status='success').exists)()
        if existing: return {'status': 'success', 'message': '已计算'}
        chip_perf = await sync_to_async(StockCyqPerf.objects.filter(stock=stock, trade_time=trade_date).first)()
        if not chip_perf: return {'status': 'failed', 'error': '无筹码性能数据'}
        chips_data = await sync_to_async(list)(chips_model.objects.filter(stock=stock, trade_time=trade_date).values('price', 'percent'))
        if not chips_data: return {'status': 'failed', 'error': '无筹码分布数据'}
        chips_df = pd.DataFrame.from_records(chips_data).astype(np.float32)
        daily_kline = await sync_to_async(daily_data_model.objects.filter(stock=stock, trade_time=trade_date).first)()
        if not daily_kline: return {'status': 'failed', 'error': '无日K线数据'}
        prev_date = trade_date - timedelta(days=1)
        prev_chips_data = await sync_to_async(list)(chips_model.objects.filter(stock=stock, trade_time=prev_date).values('price', 'percent'))
        prev_chips_df = pd.DataFrame.from_records(prev_chips_data).astype(np.float32) if prev_chips_data else pd.DataFrame()
        historical_df = await get_historical_prices_for_stock(stock_code, trade_date, ChipTaskConfig.HISTORICAL_DAYS_FOR_MA)
        historical_prices_series = historical_df['close_qfq'] if 'close_qfq' in historical_df else pd.Series(dtype=np.float32)
        historical_factors = await get_historical_chip_factors(chip_factor_model, stock, trade_date, 5)
        current_turnover = historical_df.loc[trade_date, 'turnover_rate'] if trade_date in historical_df.index else 0.0
        chip_perf_dict = {'weight_avg': chip_perf.weight_avg, 'his_high': chip_perf.his_high, 'his_low': chip_perf.his_low, 'cost_5pct': chip_perf.cost_5pct, 'cost_15pct': chip_perf.cost_15pct, 'cost_50pct': chip_perf.cost_50pct, 'cost_85pct': chip_perf.cost_85pct, 'cost_95pct': chip_perf.cost_95pct, 'winner_rate': chip_perf.winner_rate}
        daily_kline_dict = {'close': daily_kline.close_qfq, 'open': daily_kline.open_qfq, 'high': daily_kline.high_qfq, 'low': daily_kline.low_qfq, 'vol': daily_kline.vol, 'amount': daily_kline.amount, 'pct_change': daily_kline.pct_change}
        daily_basic_dict = {'turnover_rate': current_turnover}
        factors = ChipFactorCalculator.calculate_complete_factors(chip_perf_data=chip_perf_dict, chip_dist_data=chips_df, daily_basic_data=daily_basic_dict, daily_kline_data=daily_kline_dict, prev_chip_dist_data=prev_chips_df, historical_prices=historical_prices_series, historical_chip_factors=historical_factors)
        await save_chip_factors(chip_factor_model, stock, trade_date, factors)
        verify_result = await verify_chip_factor_saved(stock_code, trade_date)
        if verify_result.get('exists'):
            print(f"✅ [单日计算] 验证通过: {stock_code} {trade_date} 已保存到数据库")
        return {'status': 'success', 'message': '计算完成'}
    except Exception as e:
        logger.error(f"计算股票 {stock_code} 日期 {trade_date} 失败: {e}")
        return {'status': 'error', 'error': str(e)}

@celery_app.task(bind=True, name='tasks.chip_factor_tasks.calculate_holding_matrix_batch', queue=ChipTaskConfig.get_queue_name())
def calculate_holding_matrix_batch(self, stock_codes: List[str], start_date: str, end_date: str, market: str = None, incremental: bool = True) -> Dict:
    # [V3.1.0] 为持有矩阵新增O(1)增量探底，阻断已完成日期的无效运算
    try:
        logger.info(f"开始批量计算持有时间矩阵，股票数量: {len(stock_codes)}，增量模式: {incremental}")
        global_start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        results = {'total': len(stock_codes), 'success': 0, 'failed': 0, 'details': []}
        print(f"📋 [持有矩阵批量] 开始处理 {len(stock_codes)} 只股票")
        for stock_index, stock_code in enumerate(stock_codes):
            try:
                print(f"🔴 [持有矩阵单股] 开始处理第 {stock_index + 1}/{len(stock_codes)} 只股票: {stock_code}")
                actual_start_date_obj = global_start_date_obj
                if incremental:
                    holding_matrix_model = get_chip_holding_matrix_model_by_code(stock_code)
                    latest_record = holding_matrix_model.objects.filter(
                        stock__stock_code=stock_code,
                        calc_status='success'
                    ).order_by('-trade_time').first()
                    if latest_record:
                        next_trade_date = TradeCalendar.get_next_trade_date(latest_record.trade_time)
                        if next_trade_date:
                            actual_start_date_obj = max(global_start_date_obj, next_trade_date)
                            print(f"🔍 [增量探底] {stock_code} 持有矩阵最新成功日期: {latest_record.trade_time}, 调整为: {actual_start_date_obj}")
                if actual_start_date_obj > end_date_obj:
                    print(f"✅ [增量跳过] {stock_code} 持有矩阵已是最新，跳过计算")
                    results['success'] += 1
                    results['details'].append({'stock_code': stock_code, 'stock_index': stock_index + 1, 'status': 'success', 'processed_dates': 0})
                    continue
                result = calculate_single_stock_holding_matrix_sync(stock_code, actual_start_date_obj, end_date_obj)
                if result.get('status') == 'success':
                    results['success'] += 1
                    print(f"✅ [持有矩阵单股完成] {stock_code} 处理完成，成功 {result.get('processed_dates', 0)} 个交易日")
                else:
                    results['failed'] += 1
                    print(f"❌ [持有矩阵单股失败] {stock_code} 处理失败: {result.get('error', '未知错误')}")
                results['details'].append({'stock_code': stock_code, 'stock_index': stock_index + 1, **result})
            except Exception as e:
                results['failed'] += 1
                print(f"❌ [持有矩阵单股异常] {stock_code} 处理异常: {e}")
                results['details'].append({'stock_code': stock_code, 'stock_index': stock_index + 1, 'status': 'error', 'error': str(e), 'processed_dates': 0})
        return results
    except Exception as e:
        logger.error(f"批量计算持有时间矩阵失败: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=ChipTaskConfig.RETRY_DELAY)

@celery_app.task(bind=True, name='tasks.chip_factor_tasks.schedule_holding_matrix_calculation', queue=ChipTaskConfig.get_queue_name())
def schedule_holding_matrix_calculation(self,stock_codes: Optional[List[str]] = None,start_date_str: Optional[str] = None,end_date_str: Optional[str] = None) -> Dict:
    """
    调度持有时间矩阵计算任务
    Args:
        stock_codes: 股票代码列表，None表示全市场
        start_date_str: 开始日期 (YYYYMMDD)，None表示默认起始
        end_date_str: 结束日期 (YYYYMMDD)，None表示最近交易日
    Returns:
        Dict: 调度结果
    """
    try:
        logger.info(f"开始调度持有时间矩阵计算任务")
        # 解析日期
        if start_date_str:
            start_date = parse_date(start_date_str)
        else:
            start_date = parse_date(ChipTaskConfig.DEFAULT_START_DATE)
        if end_date_str:
            end_date = parse_date(end_date_str)
        else:
            end_date = get_last_trade_date()
        logger.info(f"持有矩阵计算日期范围: {start_date} 到 {end_date}")
        # 获取股票列表
        if stock_codes is None:
            cache_manager = CacheManager()
            stock_dao = StockBasicInfoDao(cache_manager)
            # 同步调用异步方法
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                stock_list = loop.run_until_complete(stock_dao.get_stock_list())
                stock_codes = [stock.stock_code for stock in stock_list]
            finally:
                loop.close()
        logger.info(f"需要计算持有矩阵的股票数量: {len(stock_codes)}")
        # 按市场分组
        market_groups = {}
        for code in stock_codes:
            market = get_market_from_code(code)
            market_groups.setdefault(market, []).append(code)
        total_tasks = 0
        # 为每个市场的股票创建任务
        for market, codes in market_groups.items():
            # 分批处理
            for i in range(0, len(codes), ChipTaskConfig.BATCH_SIZE_BULK):
                batch_codes = codes[i:i + ChipTaskConfig.BATCH_SIZE_BULK]
                # 创建计算任务
                task = calculate_holding_matrix_batch.delay(
                    stock_codes=batch_codes,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    market=market
                )
                total_tasks += 1
                logger.debug(f"创建持有矩阵批量计算任务 {task.id}: {len(batch_codes)} 只股票")
        result = {
            'status': 'scheduled',
            'total_tasks': total_tasks,
            'mode': 'stock_batch',
            'total_stocks': len(stock_codes),
            'date_range': f"{start_date} - {end_date}"
        }
        print(f"✅ [调度完成] 持有矩阵任务调度完成: {result}")
        return result
    except Exception as e:
        logger.error(f"调度持有矩阵计算任务失败: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=ChipTaskConfig.RETRY_DELAY)

def schedule_single_stock_comprehensive_calculation(
    stock_code: str,
    start_date_str: str = None,
    end_date_str: str = None
) -> List[str]:
    """
    调度单只股票的综合计算（命令行调用）
    Returns:
        List[str]: 任务ID列表
    """
    if start_date_str is None:
        start_date_str = ChipTaskConfig.DEFAULT_START_DATE
    if end_date_str is None:
        end_date_str = get_last_trade_date().strftime('%Y%m%d')
    task_ids = []
    # 调度筹码因子计算
    chip_task = calculate_chip_factors_batch.delay(
        stock_codes=[stock_code],
        start_date=start_date_str,
        end_date=end_date_str
    )
    task_ids.append(chip_task.id)
    # 调度持有矩阵计算
    holding_task = calculate_holding_matrix_batch.delay(
        stock_codes=[stock_code],
        start_date=start_date_str,
        end_date=end_date_str
    )
    task_ids.append(holding_task.id)
    return task_ids

# ========== 监控和状态检查任务 ==========
@celery_app.task(bind=True, name='tasks.chip_factor_tasks.check_chip_factor_status',queue=ChipTaskConfig.get_queue_name(), priority=ChipTaskConfig.PRIORITY_LOW)
def check_chip_factor_status(date_str: Optional[str] = None,market: Optional[str] = None) -> Dict:
    """
    检查筹码因子计算状态
    Args:
        date_str: 日期 (YYYYMMDD)，None表示最近交易日
        market: 市场标识，None表示全市场
    Returns:
        Dict: 状态统计
    """
    try:
        if date_str:
            check_date = parse_date(date_str)
        else:
            check_date = get_last_trade_date()
        # 根据市场获取模型
        if market == 'SZ':
            models = [ChipFactorSZ]
        elif market == 'SH':
            models = [ChipFactorSH]
        elif market == 'CY':
            models = [ChipFactorCY]
        elif market == 'KC':
            models = [ChipFactorKC]
        elif market == 'BJ':
            models = [ChipFactorBJ]
        else:
            models = [ChipFactorSZ, ChipFactorSH, ChipFactorCY, ChipFactorKC, ChipFactorBJ]
        total_count = 0
        success_count = 0
        pending_count = 0
        failed_count = 0
        for model in models:
            count = model.objects.filter(trade_time=check_date).count()
            success = model.objects.filter(trade_time=check_date, calc_status='success').count()
            pending = model.objects.filter(trade_time=check_date, calc_status='pending').count()
            failed = model.objects.filter(trade_time=check_date, calc_status='failed').count()
            total_count += count
            success_count += success
            pending_count += pending
            failed_count += failed
        # 获取总股票数
        total_stocks = StockInfo.objects.filter(list_status='L').count()
        result = {
            'date': check_date.strftime('%Y%m%d'),
            'total_stocks': total_stocks,
            'calculated_stocks': total_count,
            'success': success_count,
            'pending': pending_count,
            'failed': failed_count,
            'completion_rate': round(success_count / total_stocks * 100, 2) if total_stocks > 0 else 0
        }
        logger.info(f"筹码因子状态检查: {result}")
        return result
    except Exception as e:
        logger.error(f"检查筹码因子状态失败: {e}")
        return {'status': 'error', 'error': str(e)}

def check_holding_matrix_status(stock_code: str, start_date: date, end_date: date) -> Dict:
    """
    检查单个股票在指定日期范围内的持有矩阵计算状态
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    Returns:
        Dict: 状态信息
    """
    try:
        from utils.model_helpers import get_chip_holding_matrix_model_by_code
        HoldingMatrixModel = get_chip_holding_matrix_model_by_code(stock_code)
        # 查询成功计算的记录数
        success_count = HoldingMatrixModel.objects.filter(
            stock__stock_code=stock_code,
            trade_time__gte=start_date,
            trade_time__lte=end_date,
            calc_status='success'
        ).count()
        # 查询总交易日数
        trade_dates = TradeCalendar.get_trade_dates_between(start_date, end_date)
        total_days = len(trade_dates) if trade_dates else 0
        return {
            'stock_code': stock_code,
            'success_count': success_count,
            'total_days': total_days,
            'completion_rate': success_count / total_days if total_days > 0 else 0,
            'has_data': success_count > 0
        }
    except Exception as e:
        print(f"❌ [检查持有矩阵状态失败] {stock_code}: {e}")
        return {
            'stock_code': stock_code,
            'success_count': 0,
            'total_days': 0,
            'completion_rate': 0,
            'has_data': False
        }

# ========== 定时任务 ==========
@celery_app.task(bind=True, name='tasks.chip_factor_tasks.daily_chip_factor_update',queue=ChipTaskConfig.get_queue_name(), priority=ChipTaskConfig.PRIORITY_HIGH)
def daily_chip_factor_update(self) -> Dict:
    """
    每日筹码因子更新任务
    在筹码数据更新后（18-19点）运行
    """
    try:
        # 获取最近交易日
        trade_date = get_last_trade_date()
        date_str = trade_date.strftime('%Y%m%d')
        logger.info(f"开始每日筹码因子更新: {date_str}")
        # 检查是否已计算
        status = check_chip_factor_status(date_str)
        if status.get('completion_rate', 0) > 95:
            logger.info(f"日期 {date_str} 筹码因子已基本计算完成，跳过")
            return {
                'status': 'skipped',
                'message': '已基本计算完成',
                **status
            }
        # 调度计算任务
        task = schedule_chip_factor_calculation.delay(
            stock_codes=None,  # 全市场
            start_date_str=date_str,
            end_date_str=date_str,
            batch_mode=True
        )
        return {
            'status': 'scheduled',
            'task_id': task.id,
            'date': date_str,
            'message': '已调度每日更新任务'
        }
    except Exception as e:
        logger.error(f"每日筹码因子更新失败: {e}")
        return {'status': 'error', 'error': str(e)}

@celery_app.task(bind=True, name='tasks.chip_factor_tasks.weekly_chip_factor_maintenance',queue=ChipTaskConfig.get_queue_name(), priority=ChipTaskConfig.PRIORITY_LOW)
def weekly_chip_factor_maintenance(self) -> Dict:
    """
    每周筹码因子维护任务
    补充历史数据、清理异常数据等
    """
    try:
        logger.info("开始每周筹码因子维护")
        # 1. 检查最近30天数据完整性
        end_date = get_last_trade_date()
        start_date = end_date - timedelta(days=30)
        incomplete_dates = []
        current_date = start_date
        while current_date <= end_date:
            status = check_chip_factor_status(current_date.strftime('%Y%m%d'))
            if status.get('completion_rate', 0) < 90:
                incomplete_dates.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
        # 2. 调度补充计算
        if incomplete_dates:
            logger.info(f"发现不完整日期: {incomplete_dates}")
            for date_str in incomplete_dates:
                schedule_chip_factor_calculation.delay(
                    stock_codes=None,
                    start_date_str=date_str,
                    end_date_str=date_str,
                    batch_mode=True
                )
        # 3. 清理失败记录（保留7天内的）
        cleanup_date = end_date - timedelta(days=7)
        cleanup_count = 0
        for model in [ChipFactorSZ, ChipFactorSH, ChipFactorCY, ChipFactorKC, ChipFactorBJ]:
            count, _ = model.objects.filter(
                calc_status='failed',
                trade_time__lt=cleanup_date
            ).delete()
            cleanup_count += count
        result = {
            'status': 'completed',
            'incomplete_dates': len(incomplete_dates),
            'dates': incomplete_dates,
            'cleaned_records': cleanup_count,
            'message': '每周维护完成'
        }
        logger.info(f"每周筹码因子维护完成: {result}")
        return result
    except Exception as e:
        logger.error(f"每周筹码因子维护失败: {e}")
        return {'status': 'error', 'error': str(e)}

@celery_app.task(bind=True, name='tasks.chip_factor_tasks.schedule_energy_field_analysis',queue=ChipTaskConfig.get_queue_name())
def schedule_energy_field_analysis(self, start_date_str: str, end_date_str: str) -> Dict:
    """
    专门调度能量场分析任务
    """
    try:
        start_date = parse_date(start_date_str)
        end_date = parse_date(end_date_str)
        # 获取全市场股票
        cache_manager = CacheManager()
        stock_dao = StockBasicInfoDao(cache_manager)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            stock_list = loop.run_until_complete(stock_dao.get_stock_list())
            stock_codes = [stock.stock_code for stock in stock_list]
        finally:
            loop.close()
        # 分批调度
        batch_size = ChipTaskConfig.BATCH_SIZE_BULK
        total_batches = (len(stock_codes) + batch_size - 1) // batch_size
        task_ids = []
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i + batch_size]
            task = calculate_energy_field_batch.delay(
                stock_codes=batch_codes,
                start_date=start_date_str,
                end_date=end_date_str
            )
            task_ids.append(task.id)
            print(f"📤 [能量场调度] 批次 {i//batch_size + 1}/{total_batches}: {len(batch_codes)} 只股票")
        return {
            'status': 'scheduled',
            'task_count': len(task_ids),
            'task_ids': task_ids,
            'total_stocks': len(stock_codes),
        }
    except Exception as e:
        logger.error(f"调度能量场分析失败: {e}")
        raise self.retry(exc=e, countdown=60)

@celery_app.task(bind=True, name='tasks.chip_factor_tasks.calculate_energy_field_batch',queue=ChipTaskConfig.get_queue_name())
def calculate_energy_field_batch(self, stock_codes: List[str], start_date: str, end_date: str, incremental: bool = True) -> Dict:
    # [V3.1.0] 能量场分析批处理增加O(1)增量过滤逻辑，依据能量场特有字段判断断点
    try:
        global_start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        results = {'total': len(stock_codes), 'success': 0, 'failed': 0, 'details': []}
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cm = CacheManager()
        try:
            for idx, stock_code in enumerate(stock_codes):
                try:
                    print(f"🔴 [能量场计算] 处理 {idx+1}/{len(stock_codes)}: {stock_code}")
                    actual_start_date_obj = global_start_date_obj
                    if incremental:
                        holding_matrix_model = get_chip_holding_matrix_model_by_code(stock_code)
                        # 查询能量场特有字段不为默认值的最新日期
                        latest_record = holding_matrix_model.objects.filter(
                            stock__stock_code=stock_code,
                            calc_status='success',
                            analysis_method='energy_field_v2'
                        ).order_by('-trade_time').first()
                        if latest_record:
                            next_trade_date = TradeCalendar.get_next_trade_date(latest_record.trade_time)
                            if next_trade_date:
                                actual_start_date_obj = max(global_start_date_obj, next_trade_date)
                    if actual_start_date_obj > end_date_obj:
                        print(f"✅ [增量跳过] {stock_code} 能量场数据已是最新")
                        results['success'] += 1
                        continue
                    trade_dates = TradeCalendar.get_trade_dates_between(actual_start_date_obj, end_date_obj)
                    if not trade_dates:
                        continue
                    date_results = loop.run_until_complete(
                        process_energy_field_for_stock(stock_code, trade_dates, cm)
                    )
                    processed_dates = date_results['processed_dates']
                    if processed_dates > 0:
                        results['success'] += 1
                        results['details'].append({'stock_code': stock_code, 'status': 'success', 'processed_dates': processed_dates})
                    else:
                        results['failed'] += 1
                except Exception as e:
                    results['failed'] += 1
                    print(f"❌ [能量场股票错误] {stock_code}: {e}")
        finally:
            loop.close()
        return results
    except Exception as e:
        logger.error(f"批量计算能量场失败: {e}")
        raise self.retry(exc=e, countdown=60)

async def process_energy_field_for_stock(stock_code: str, trade_dates: List[date], cm: CacheManager) -> Dict:
    """处理单只股票的能量场计算"""
    processed_dates = 0
    # 创建AdvancedChipDynamicsService实例
    service = AdvancedChipDynamicsService(market_type=get_market_from_code(stock_code))
    for trade_date in trade_dates:
        try:
            # 分析筹码动态（包含能量场）
            realtime_dao = StockRealtimeDAO(cm)
            tick_data = await realtime_dao.get_daily_real_ticks(stock_code, trade_date) 
            dynamics_result = await service.analyze_chip_dynamics_daily(
                stock_code=stock_code,
                trade_date=trade_date.strftime('%Y-%m-%d'),
                lookback_days=20,
                tick_data=tick_data
            )
            if dynamics_result.get('analysis_status') == 'success':
                # 更新持有矩阵记录的能量场字段
                HoldingMatrixModel = get_chip_holding_matrix_model_by_code(stock_code)
                # 获取股票
                from stock_models.stock_basic import StockInfo
                stock = await sync_to_async(StockInfo.objects.filter(stock_code=stock_code).first)()
                if stock:
                    # 获取或创建记录
                    record, created = await sync_to_async(HoldingMatrixModel.objects.get_or_create)(
                        stock=stock,
                        trade_time=trade_date,
                        defaults={'calc_status': 'pending'}
                    )
                    # 只更新能量场相关字段，不覆盖其他计算
                    game_energy = dynamics_result.get('game_energy_result', {})
                    if game_energy:
                        record.absorption_energy = game_energy.get('absorption_energy', 0.0)
                        record.distribution_energy = game_energy.get('distribution_energy', 0.0)
                        record.net_energy_flow = game_energy.get('net_energy_flow', 0.0)
                        record.game_intensity = game_energy.get('game_intensity', 0.0)
                        record.breakout_potential = game_energy.get('breakout_potential', 0.0)
                        record.energy_concentration = game_energy.get('energy_concentration', 0.0)
                        record.fake_distribution_flag = game_energy.get('fake_distribution_flag', False)
                        record.key_battle_zones = game_energy.get('key_battle_zones', [])
                        # 标记为能量场计算
                        record.analysis_method = 'energy_field_v2'
                        await sync_to_async(record.save)()
                        processed_dates += 1
        except Exception as e:
            print(f"⚠️ [能量场日期错误] {stock_code} {trade_date}: {e}")
            continue
    return {'processed_dates': processed_dates}

# ========== 工具函数 ==========
def schedule_comprehensive_calculation(stock_codes: List[str], start_date: date, end_date: date, batch_mode: bool = True, incremental: bool = True) -> Dict:
    # [V3.1.0] 综合计算调度器注入incremental参数，贯穿全链路增量控制
    print(f"🚀 [综合计算V3] 开始调度 {len(stock_codes)} 只股票，增量模式: {incremental}")
    print(f"📅 [日期范围] {start_date} 到 {end_date}")
    market_groups = {}
    for code in stock_codes:
        market = get_market_from_code(code)
        market_groups.setdefault(market, []).append(code)
    total_tasks = 0
    task_details = []
    for market, codes in market_groups.items():
        print(f"📊 [市场分组] {market}: {len(codes)} 只股票")
        for i in range(0, len(codes), ChipTaskConfig.BATCH_SIZE_BULK):
            batch_codes = codes[i:i + ChipTaskConfig.BATCH_SIZE_BULK]
            task_chain = chain(
                calculate_holding_matrix_batch.s(
                    stock_codes=batch_codes,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    market=market,
                    incremental=incremental
                ),
                calculate_energy_field_batch.si(
                    stock_codes=batch_codes,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    incremental=incremental
                ),
                calculate_chip_factors_batch.si(
                    stock_codes=batch_codes,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    market=market,
                    incremental=incremental
                )
            )
            chain_result = task_chain.delay()
            total_tasks += 1
            task_details.append({
                'task_id': chain_result.id,
                'market': market,
                'batch_index': i // ChipTaskConfig.BATCH_SIZE_BULK + 1,
                'stock_count': len(batch_codes)
            })
            if total_tasks % 10 == 0:
                time.sleep(0.5)
    print(f"✅ [综合计算V3] 共创建 {total_tasks} 个链式任务")
    return {
        'status': 'scheduled',
        'total_tasks': total_tasks,
        'mode': 'comprehensive_v3',
        'calculation_flow': 'holding_matrix -> energy_field -> chip_factors',
        'total_stocks': len(stock_codes),
        'date_range': f"{start_date} - {end_date}"
    }

def schedule_energy_only_calculation(
    stock_codes: List[str], 
    start_date: date, 
    end_date: date
) -> Dict:
    """
    仅调度能量场分析
    """
    print(f"⚡ [能量场专用] 开始调度 {len(stock_codes)} 只股票的能量场计算")
    # 按市场分组
    market_groups = {}
    for code in stock_codes:
        market = get_market_from_code(code)
        market_groups.setdefault(market, []).append(code)
    total_tasks = 0
    for market, codes in market_groups.items():
        # 分批处理
        for i in range(0, len(codes), ChipTaskConfig.BATCH_SIZE_BULK):
            batch_codes = codes[i:i + ChipTaskConfig.BATCH_SIZE_BULK]
            # 创建能量场计算任务
            task = calculate_energy_field_batch.delay(
                stock_codes=batch_codes,
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d')
            )
            total_tasks += 1
            print(f"📤 [能量场任务] 创建任务 {task.id}: {market}市场 {len(batch_codes)} 只股票")
    return {
        'status': 'scheduled',
        'total_tasks': total_tasks,
        'mode': 'energy_only',
        'total_stocks': len(stock_codes),
        'date_range': f"{start_date} - {end_date}"
    }

def schedule_single_stock_calculation(stock_code: str,start_date_str: str = None,end_date_str: str = None) -> str:
    """
    调度单只股票计算（命令行调用）
    """
    if start_date_str is None:
        start_date_str = ChipTaskConfig.DEFAULT_START_DATE
    if end_date_str is None:
        end_date_str = get_last_trade_date().strftime('%Y%m%d')
    task = calculate_chip_factors_batch.delay(
        stock_codes=[stock_code],
        start_date=start_date_str,
        end_date=end_date_str
    )
    return task.id