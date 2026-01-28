# tasks\chip_factor_tasks.py
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
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 导入模型和工具
from stock_models.chip_factors import (
    ChipFactorSZ, ChipFactorSH, ChipFactorCY, 
    ChipFactorKC, ChipFactorBJ, 
)
from stock_models.chip import StockCyqPerf
from stock_models.index import TradeCalendar
from utils.model_helpers import (
    get_cyq_chips_model_by_code,
    get_daily_data_model_by_code,
    get_chip_factor_model_by_code,
    get_chip_holding_matrix_model_by_code,
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
    BATCH_SIZE_BULK = 50    # 批量股票每批数量
    
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
    self,  stock_codes: Optional[List[str]] = None, start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = None, batch_mode: bool = True
) -> Dict:
    """
    调度筹码因子计算任务
    
    Args:
        stock_codes: 股票代码列表，None表示全市场
        start_date_str: 开始日期 (YYYYMMDD)，None表示默认起始
        end_date_str: 结束日期 (YYYYMMDD)，None表示最近交易日
        batch_mode: 是否批量模式（True=按股票分批，False=按日期分批）
    
    Returns:
        Dict: 调度结果
    """
    try:
        logger.info(f"开始调度筹码因子计算任务")
        # 解析日期
        if start_date_str:
            start_date = parse_date(start_date_str)
        else:
            start_date = parse_date(ChipTaskConfig.DEFAULT_START_DATE)
        if end_date_str:
            end_date = parse_date(end_date_str)
        else:
            end_date = get_last_trade_date()
        logger.info(f"日期范围: {start_date} 到 {end_date}")
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
        logger.info(f"需要计算的股票数量: {len(stock_codes)}")
        # 根据模式调度
        if batch_mode:
            # 按股票分批
            result = schedule_by_stock_batch(stock_codes, start_date, end_date)
        else:
            # 按日期分批
            result = schedule_by_date_batch(stock_codes, start_date, end_date)
        logger.info(f"筹码因子计算任务调度完成: {result}")
        return result
        
    except Exception as e:
        logger.error(f"调度筹码因子计算任务失败: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=ChipTaskConfig.RETRY_DELAY)

def schedule_by_stock_batch(stock_codes: List[str], start_date: date, end_date: date) -> Dict:
    """按股票分批调度"""
    total_tasks = 0
    print(f"📋 [调度开始] 按股票分批调度 {len(stock_codes)} 只股票")
    print(f"📅 [调度日期] 日期范围: {start_date} 到 {end_date}")
    # 按市场分组
    market_groups = {}
    for code in stock_codes:
        market = get_market_from_code(code)
        market_groups.setdefault(market, []).append(code)
    print(f"📊 [调度分组] 按市场分组: {len(market_groups)} 个市场")
    for market, codes in market_groups.items():
        print(f"📊 [调度市场] {market}: {len(codes)} 只股票")
    # 为每个市场的股票创建任务
    for market, codes in market_groups.items():
        # 分批处理
        for i in range(0, len(codes), ChipTaskConfig.BATCH_SIZE_BULK):
            batch_codes = codes[i:i + ChipTaskConfig.BATCH_SIZE_BULK]
            # 创建计算任务
            task = calculate_chip_factors_batch.delay(stock_codes=batch_codes, start_date=start_date.strftime('%Y%m%d'), end_date=end_date.strftime('%Y%m%d'), market=market)
            total_tasks += 1
            print(f"📤 [调度任务] 创建任务 {task.id}: {market}市场 {len(batch_codes)} 只股票")
            print(f"📤 [调度批次] 批次 {i//ChipTaskConfig.BATCH_SIZE_BULK + 1}: 股票 {batch_codes[:3]}{'...' if len(batch_codes) > 3 else ''}")
    print(f"✅ [调度完成] 共创建 {total_tasks} 个任务")
    return {'status': 'scheduled', 'total_tasks': total_tasks, 'mode': 'stock_batch', 'total_stocks': len(stock_codes), 'date_range': f"{start_date} - {end_date}"}

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
def calculate_chip_factors_batch(self, stock_codes: List[str], start_date: str, end_date: str, market: str = None) -> Dict:
    """
    批量计算多只股票的筹码因子（按股票循环）
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        market: 市场标识（可选，自动识别）
    Returns:
        Dict: 计算结果
    """
    try:
        logger.info(f"开始批量计算筹码因子，股票数量: {len(stock_codes)}")
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        results = {
            'total': len(stock_codes),
            'success': 0,
            'failed': 0,
            'details': []
        }
        print(f"📋 [批量任务] 开始处理 {len(stock_codes)} 只股票")
        print(f"📋 [批量任务] 日期范围: {start_date_obj} 到 {end_date_obj}")
        # 改为按股票顺序处理，完成一只股票的所有日期后再处理下一只
        for stock_index, stock_code in enumerate(stock_codes):
            try:
                print(f"🔴 [股票处理] 开始处理第 {stock_index + 1}/{len(stock_codes)} 只股票: {stock_code}")
                # 检查该股票的持有矩阵是否已计算（仅检查，不调度）
                print(f"🔍 [股票检查] 检查 {stock_code} 的持有矩阵计算状态...")
                HoldingMatrixModel = get_chip_holding_matrix_model_by_code(stock_code)
                # 查询该股票在日期范围内是否有成功计算的持有矩阵
                date_range_count = HoldingMatrixModel.objects.filter(
                    stock__stock_code=stock_code,
                    trade_time__gte=start_date_obj,
                    trade_time__lte=end_date_obj,
                    calc_status='success'
                ).count()
                if date_range_count == 0:
                    print(f"⚠️ [股票检查] {stock_code} 在日期范围内无成功计算的持有矩阵，筹码因子将使用默认值")
                else:
                    print(f"✅ [股票检查] {stock_code} 已有 {date_range_count} 天的持有矩阵数据")
                # 调用同步版本的单个股票计算函数
                result = calculate_single_stock_chip_factors_sync(
                    stock_code,
                    start_date_obj,
                    end_date_obj
                )
                # 收集结果
                if result.get('status') == 'success':
                    results['success'] += 1
                    processed_dates = result.get('processed_dates', 0)
                    print(f"✅ [股票完成] {stock_code} 处理完成，成功 {processed_dates} 个交易日")
                else:
                    results['failed'] += 1
                    error_msg = result.get('error', '未知错误')
                    print(f"❌ [股票失败] {stock_code} 处理失败: {error_msg}")
                    logger.warning(f"股票 {stock_code} 计算失败: {error_msg}")
                results['details'].append({
                    'stock_code': stock_code,
                    'stock_index': stock_index + 1,
                    **result
                })
                # 每完成5只股票打印一次进度
                if (stock_index + 1) % 5 == 0:
                    print(f"📊 [进度报告] 已完成 {stock_index + 1}/{len(stock_codes)} 只股票")
                    print(f"📊 [进度报告] 成功: {results['success']}, 失败: {results['failed']}")
            except Exception as e:
                results['failed'] += 1
                print(f"❌ [股票异常] {stock_code} 处理异常: {e}")
                logger.error(f"股票 {stock_code} 计算异常: {e}")
                results['details'].append({
                    'stock_code': stock_code,
                    'stock_index': stock_index + 1,
                    'status': 'error',
                    'error': str(e),
                    'processed_dates': 0
                })
        logger.info(f"批量计算完成: 成功 {results['success']}, 失败 {results['failed']}")
        print(f"✅ [批量任务完成] 总计: 成功 {results['success']}, 失败 {results['failed']}")
        return results
    except Exception as e:
        logger.error(f"批量计算筹码因子失败: {e}", exc_info=True)
        print(f"❌ [批量任务异常] {e}")
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
            print(f"⏱️ [单股耗时] {stock_code} 处理耗时: {elapsed_time:.2f}秒")
            return result
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"同步计算股票 {stock_code} 失败: {e}")
        print(f"❌ [单股异常] {stock_code}: {e}")
        return {'status': 'error', 'error': str(e), 'processed_dates': 0}

async def calculate_single_stock_chip_factors_async(stock_code: str, start_date: date, end_date: date) -> Dict:
    """异步版本的单个股票计算函数（按股票循环）"""
    try:
        logger.debug(f"开始计算股票 {stock_code} 的筹码因子")
        from stock_models.index import TradeCalendar
        from asgiref.sync import sync_to_async
        print(f"🔴 [单股异步开始] {stock_code} {start_date} 到 {end_date}")
        # 注意：不再在这里调度持有矩阵计算，因为链式调度已经确保了先后顺序
        # 检查持有矩阵数据是否已存在
        HoldingMatrixModel = get_chip_holding_matrix_model_by_code(stock_code)
        holding_count = await sync_to_async(HoldingMatrixModel.objects.filter(
            stock__stock_code=stock_code,
            trade_time__gte=start_date,
            trade_time__lte=end_date,
            calc_status='success'
        ).count)()
        if holding_count == 0:
            print(f"⚠️ [单股检查] {stock_code} 在日期范围内无持有矩阵数据，相关字段将使用默认值")
        else:
            print(f"✅ [单股检查] {stock_code} 已有 {holding_count} 天的持有矩阵数据")
        # 后续步骤：计算筹码因子（这部分逻辑与原方法相同）
        # 获取对应的模型
        chip_factor_model = get_chip_factor_model_by_code(stock_code)
        chips_model = get_cyq_chips_model_by_code(stock_code)
        daily_data_model = get_daily_data_model_by_code(stock_code)
        # 获取股票基本信息
        stock = await sync_to_async(StockInfo.objects.filter(stock_code=stock_code).first)()
        if not stock:
            print(f"❌ [单股错误] {stock_code} 股票不存在")
            return {'status': 'failed', 'error': f'未找到股票 {stock_code}', 'processed_dates': 0}
        print(f"📊 [单股信息] 股票: {stock.stock_code} - {stock.stock_name}")
        print(f"📊 [单股信息] 因子模型: {chip_factor_model.__name__}")
        # 获取历史价格数据用于计算MA
        print(f"📊 [单股数据] 获取历史价格数据...")
        historical_prices = await get_historical_prices_for_stock(stock_code, end_date, ChipTaskConfig.HISTORICAL_DAYS_FOR_MA)
        if historical_prices.empty:
            print(f"❌ [单股错误] {stock_code} 历史价格数据不足")
            return {'status': 'failed', 'error': f'股票 {stock_code} 历史价格数据不足', 'processed_dates': 0}
        print(f"📊 [单股数据] 历史价格: {len(historical_prices)} 条")
        # 获取日期范围内的所有交易日（异步调用）
        get_dates_between_func = sync_to_async(TradeCalendar.get_trade_dates_between, thread_sensitive=True)
        trade_dates = await get_dates_between_func(start_date, end_date)
        if not trade_dates:
            print(f"⚠️ [单股警告] {stock_code} 日期范围内无交易日: {start_date} 到 {end_date}")
            return {'status': 'failed', 'error': '日期范围内无交易日', 'processed_dates': 0}
        print(f"📅 [单股日期] {stock_code} 交易日: {len(trade_dates)} 天, 从 {trade_dates[0]} 到 {trade_dates[-1]}")
        processed_dates = 0
        saved_dates = []
        failed_dates = []
        date_progress_interval = max(1, len(trade_dates) // 10)  # 每10%打印一次进度
        # 按日期循环处理当前股票
        for date_index, current_date in enumerate(trade_dates):
            try:
                # 打印进度
                if date_index % date_progress_interval == 0:
                    progress = (date_index + 1) / len(trade_dates) * 100
                    print(f"📊 [单股进度] {stock_code} 进度: {progress:.1f}% ({date_index + 1}/{len(trade_dates)})")
                # 检查是否已计算
                existing = await sync_to_async(chip_factor_model.objects.filter(stock=stock, trade_time=current_date, calc_status='success').exists)()
                if existing:
                    continue
                # 获取数据
                chip_perf = await sync_to_async(StockCyqPerf.objects.filter(stock=stock, trade_time=current_date).first)()
                if not chip_perf:
                    continue
                # 获取筹码分布数据
                chips_data = await sync_to_async(list)(chips_model.objects.filter(stock=stock, trade_time=current_date).values('price', 'percent'))
                if not chips_data:
                    continue
                chips_df = pd.DataFrame(chips_data)
                # 获取日K线数据
                daily_kline = await sync_to_async(daily_data_model.objects.filter(stock=stock, trade_time=current_date).first)()
                if not daily_kline:
                    continue
                # 获取前一日筹码数据
                get_offset_func = sync_to_async(TradeCalendar.get_trade_date_offset, thread_sensitive=True)
                prev_date = await get_offset_func(current_date, -1)
                if prev_date:
                    prev_chips_data = await sync_to_async(list)(chips_model.objects.filter(stock=stock, trade_time=prev_date).values('price', 'percent'))
                    prev_chips_df = pd.DataFrame(list(prev_chips_data)) if prev_chips_data else pd.DataFrame()
                else:
                    prev_chips_df = pd.DataFrame()
                # 获取历史因子
                historical_factors = await get_historical_chip_factors(chip_factor_model, stock, current_date, 5)
                # 准备数据字典
                chip_perf_dict = {'weight_avg': chip_perf.weight_avg, 'his_high': chip_perf.his_high, 'his_low': chip_perf.his_low, 'cost_5pct': chip_perf.cost_5pct, 'cost_15pct': chip_perf.cost_15pct, 'cost_50pct': chip_perf.cost_50pct, 'cost_85pct': chip_perf.cost_85pct, 'cost_95pct': chip_perf.cost_95pct, 'winner_rate': chip_perf.winner_rate}
                daily_kline_dict = {'close': daily_kline.close_qfq, 'open': daily_kline.open_qfq, 'high': daily_kline.high_qfq, 'low': daily_kline.low_qfq, 'vol': daily_kline.vol, 'amount': daily_kline.amount, 'pct_change': daily_kline.pct_change}
                # 计算因子
                factors = ChipFactorCalculator.calculate_complete_factors(chip_perf_data=chip_perf_dict, chip_dist_data=chips_df, daily_basic_data={}, daily_kline_data=daily_kline_dict, prev_chip_dist_data=prev_chips_df, historical_prices=historical_prices, historical_chip_factors=historical_factors)
                # 保存到数据库
                await save_chip_factors(chip_factor_model, stock, current_date, factors)
                # 验证保存结果
                verify_result = await verify_chip_factor_saved(stock_code, current_date)
                if verify_result.get('exists'):
                    processed_dates += 1
                    saved_dates.append(current_date)
                else:
                    failed_dates.append(current_date)
            except Exception as e:
                logger.warning(f"股票 {stock_code} 日期 {current_date} 计算失败: {e}")
                failed_dates.append(current_date)
        # 打印最终结果
        print(f"✅ [单股完成] {stock_code} 处理完成")
        print(f"📊 [单股统计] 成功: {len(saved_dates)} 天, 失败: {len(failed_dates)} 天")
        if saved_dates:
            print(f"📊 [单股日期] 最早成功日期: {min(saved_dates)}, 最晚成功日期: {max(saved_dates)}")
        if failed_dates and len(failed_dates) <= 10:
            print(f"⚠️ [单股失败] 失败日期: {failed_dates}")
        logger.info(f"股票 {stock_code} 计算完成，成功 {len(saved_dates)} 个交易日，失败 {len(failed_dates)} 个交易日")
        return {'status': 'success', 'processed_dates': processed_dates, 'saved_dates': len(saved_dates), 'failed_dates': len(failed_dates), 'date_range': f"{start_date} - {end_date}"}
    except Exception as e:
        logger.error(f"计算股票 {stock_code} 筹码因子失败: {e}", exc_info=True)
        print(f"❌ [单股异常] {stock_code}: {e}")
        return {'status': 'error', 'error': str(e), 'processed_dates': 0}

def calculate_single_stock_holding_matrix_sync(stock_code: str, start_date: date, end_date: date) -> Dict:
    """同步版本的单个股票持有矩阵计算函数（按股票循环）版本：重构适配AdvancedChipDynamicsService"""
    try:
        print(f"🔴 [持有矩阵单股开始] 开始处理股票 {stock_code}")
        print(f"📅 [持有矩阵单股日期] 日期范围: {start_date} 到 {end_date}")
        # 创建事件循环用于异步调用
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(calculate_single_stock_holding_matrix_async(stock_code, start_date, end_date))
            return result
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"同步计算股票 {stock_code} 持有矩阵失败: {e}")
        print(f"❌ [持有矩阵单股异常] {stock_code}: {e}")
        return {'status': 'error', 'error': str(e), 'processed_dates': 0}

async def calculate_single_stock_holding_matrix_async(stock_code: str, start_date: date, end_date: date) -> Dict:
    """异步版本的单个股票持有矩阵计算函数（使用AdvancedChipDynamicsService）版本：重构适配AdvancedChipDynamicsService"""
    try:
        logger.info(f"开始计算股票 {stock_code} 的持有时间矩阵（使用AdvancedChipDynamicsService）")
        from services.chip_dynamics_service import AdvancedChipDynamicsService
        from utils.model_helpers import get_chip_holding_matrix_model_by_code
        # 获取持有矩阵模型
        holding_matrix_model = get_chip_holding_matrix_model_by_code(stock_code)
        # 获取股票基本信息
        stock = await sync_to_async(StockInfo.objects.filter(stock_code=stock_code).first)()
        if not stock:
            return {'status': 'failed', 'error': f'未找到股票 {stock_code}', 'processed_dates': 0}
        # 创建动态分析服务
        service = AdvancedChipDynamicsService(market_type=get_market_from_code(stock_code))
        processed_dates = 0
        saved_dates = []
        failed_dates = []
        # 获取日期范围内的所有交易日
        from stock_models.index import TradeCalendar
        get_dates_func = sync_to_async(TradeCalendar.get_trade_dates_between, thread_sensitive=True)
        trade_dates = await get_dates_func(start_date, end_date)
        if not trade_dates:
            print(f"⚠️ [持有矩阵] {stock_code} 日期范围内无交易日: {start_date} 到 {end_date}")
            return {'status': 'failed', 'error': '日期范围内无交易日', 'processed_dates': 0}
        print(f"📅 [持有矩阵] {stock_code} 交易日: {len(trade_dates)} 天")
        # 按日期循环处理当前股票
        for date_index, current_date in enumerate(trade_dates):
            try:
                print(f"📊 [持有矩阵进度] {stock_code} {current_date} ({date_index + 1}/{len(trade_dates)})")
                # 检查是否已计算
                existing = await sync_to_async(holding_matrix_model.objects.filter(stock=stock, trade_time=current_date, calc_status='success').exists)()
                if existing:
                    continue
                # 使用AdvancedChipDynamicsService进行动态分析
                trade_date_str = current_date.strftime('%Y-%m-%d')
                dynamics_result = await service.analyze_chip_dynamics_daily(
                    stock_code=stock_code,
                    trade_date=trade_date_str,
                    lookback_days=20
                )
                # 保存动态分析结果到数据库
                if dynamics_result.get('analysis_status') == 'success':
                    # 获取或创建记录
                    record, created = await sync_to_async(holding_matrix_model.objects.get_or_create)(
                        stock=stock,
                        trade_time=current_date,
                        defaults={'calc_status': 'pending'}
                    )
                    # 保存动态分析结果
                    save_success = record.save_dynamics_result(dynamics_result)
                    if save_success:
                        processed_dates += 1
                        saved_dates.append(current_date)
                        print(f"✅ [持有矩阵] {stock_code} {current_date} 动态分析保存成功")
                    else:
                        failed_dates.append(current_date)
                        print(f"❌ [持有矩阵] {stock_code} {current_date} 动态分析保存失败")
                else:
                    failed_dates.append(current_date)
                    print(f"⚠️ [持有矩阵] {stock_code} {current_date} 动态分析失败")
            except Exception as e:
                print(f"❌ [持有矩阵] {stock_code} {current_date} 计算失败: {e}")
                failed_dates.append(current_date)
        print(f"✅ [持有矩阵完成] {stock_code} 处理完成，成功 {len(saved_dates)} 个交易日，失败 {len(failed_dates)} 个交易日")
        return {'status': 'success', 'processed_dates': processed_dates, 'saved_dates': len(saved_dates), 'failed_dates': len(failed_dates), 'date_range': f"{start_date} - {end_date}"}
    except Exception as e:
        logger.error(f"计算股票 {stock_code} 持有矩阵失败: {e}", exc_info=True)
        print(f"❌ [持有矩阵异常] {stock_code}: {e}")
        return {'status': 'error', 'error': str(e), 'processed_dates': 0}

def calculate_holding_matrix_for_stock_sync(stock_code: str, start_date: date, end_date: date) -> Dict:
    """同步版本的股票持有矩阵计算（按股票循环）版本：重构适配AdvancedChipDynamicsService"""
    try:
        logger.info(f"开始同步计算股票 {stock_code} 的持有时间矩阵（使用AdvancedChipDynamicsService）")
        # 创建事件循环用于异步调用
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(calculate_single_stock_holding_matrix_async(stock_code, start_date, end_date))
            return result
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"同步计算股票 {stock_code} 持有矩阵失败: {e}")
        print(f"❌ [持有矩阵异常] {stock_code}: {e}")
        return {'status': 'error', 'error': str(e), 'processed_dates': 0}

async def get_historical_prices_for_stock(stock_code: str,  end_date: date,  days: int) -> pd.Series:
    """获取股票历史价格序列"""
    try:
        daily_data_model = get_daily_data_model_by_code(stock_code)
        start_date = end_date - timedelta(days=days * 2)  # 多取一些数据
        # 查询历史价格
        price_data = await sync_to_async(list)(
            daily_data_model.objects.filter(
                stock__stock_code=stock_code,
                trade_time__gte=start_date,
                trade_time__lte=end_date
            ).order_by('trade_time')
            .values('trade_time', 'close_qfq')
        )
        if not price_data:
            return pd.Series()
        # 转换为Series
        df = pd.DataFrame(price_data)
        df.set_index('trade_time', inplace=True)
        return df['close_qfq']
        
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 历史价格失败: {e}")
        return pd.Series()

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

async def save_chip_factors(chip_factor_model, stock, trade_date: date, factors: Dict):
    """保存筹码因子"""
    try:
        print(f"💾 [保存因子开始] {stock.stock_code} {trade_date}")
        print(f"💾 [模型信息] 模型类: {chip_factor_model.__name__}")
        print(f"💾 [因子数量] {len(factors)} 个因子")
        # 检查因子数据是否有效
        if not factors:
            print(f"⚠️ [保存因子] 因子数据为空，跳过保存")
            return
            
        # 先打印一些关键因子值用于调试
        key_factors = ['chip_mean', 'chip_std', 'chip_concentration_ratio', 
                      'avg_holding_days', 'short_term_chip_ratio', 'long_term_chip_ratio']
        for key in key_factors:
            if key in factors:
                print(f"💾 [因子值] {key}: {factors[key]}")
        # 检查是否已存在
        existing = await sync_to_async(chip_factor_model.objects.filter(stock=stock, trade_time=trade_date).first)()
        if existing:
            print(f"💾 [保存因子] 更新现有记录 ID: {existing.id}")
            # 更新现有记录
            for key, value in factors.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
                else:
                    print(f"⚠️ [保存因子] 字段 {key} 不在模型中，跳过")
            # 设置计算状态
            if hasattr(existing, 'calc_status'):
                existing.calc_status = 'success'
            if hasattr(existing, 'update_time'):
                existing.update_time = datetime.now()
            # 保存更新
            await sync_to_async(existing.save)()
            print(f"✅ [保存因子完成] {stock.stock_code} {trade_date} 记录更新成功")
            
        else:
            print(f"💾 [保存因子] 创建新记录")
            # 准备创建数据
            create_data = {
                'stock': stock,
                'trade_time': trade_date,
            }
            # 添加因子字段
            valid_fields = []
            for key, value in factors.items():
                # 检查字段是否在模型中
                try:
                    field = chip_factor_model._meta.get_field(key)
                    # 处理不同类型的数据
                    if isinstance(value, (int, float, str, bool, date, datetime)):
                        create_data[key] = value
                        valid_fields.append(key)
                    elif value is None:
                        create_data[key] = None
                        valid_fields.append(key)
                    else:
                        print(f"⚠️ [保存因子] 字段 {key} 值类型不支持: {type(value)}")
                except:
                    print(f"⚠️ [保存因子] 字段 {key} 不在模型中，跳过")
            # 添加计算状态
            if hasattr(chip_factor_model, 'calc_status'):
                create_data['calc_status'] = 'success'
            if hasattr(chip_factor_model, 'create_time'):
                create_data['create_time'] = datetime.now()
            if hasattr(chip_factor_model, 'update_time'):
                create_data['update_time'] = datetime.now()
            print(f"💾 [创建数据] 有效字段数: {len(valid_fields)}")
            print(f"💾 [创建数据] 字段列表: {valid_fields[:10]}{'...' if len(valid_fields) > 10 else ''}")
            try:
                # 创建新记录
                new_record = await sync_to_async(chip_factor_model.objects.create)(**create_data)
                print(f"✅ [保存因子完成] {stock.stock_code} {trade_date} 新记录创建成功, ID: {new_record.id}")
                
                # 验证记录是否真的保存了
                verify = await sync_to_async(chip_factor_model.objects.filter(id=new_record.id).exists)()
                if verify:
                    print(f"✅ [验证] 记录 {new_record.id} 已成功保存到数据库")
                else:
                    print(f"❌ [验证] 记录 {new_record.id} 未保存到数据库")
                    
            except Exception as create_error:
                print(f"❌ [保存因子失败] 创建记录时出错: {create_error}")
                print(f"💾 [创建数据详情] {create_data}")
                import traceback
                traceback.print_exc()
                raise
                
    except Exception as e:
        logger.error(f"保存筹码因子失败: {e}")
        print(f"❌ [保存因子异常] {stock.stock_code} {trade_date}: {e}")
        import traceback
        traceback.print_exc()
        raise

async def verify_chip_factor_saved(stock_code: str, trade_date: date) -> Dict:
    """验证筹码因子是否已保存到数据库"""
    try:
        from asgiref.sync import sync_to_async
        # 获取股票
        stock = await sync_to_async(StockInfo.objects.filter(stock_code=stock_code).first)()
        if not stock:
            return {'status': 'failed', 'error': '股票不存在'}
        # 获取对应的因子模型
        chip_factor_model = get_chip_factor_model_by_code(stock_code)
        print(f"🔍 [验证] 检查 {stock_code} {trade_date} 的筹码因子")
        print(f"🔍 [验证] 使用模型: {chip_factor_model.__name__}")
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
                        print(f"🔍 [验证字段] {field}: {value}")
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
    """计算单只股票单日筹码因子（异步版本）"""
    try:
        print(f"🔴 [单日计算] 开始计算 {stock_code} {trade_date}")
        # 获取模型
        chip_factor_model = get_chip_factor_model_by_code(stock_code)
        chips_model = get_cyq_chips_model_by_code(stock_code)
        daily_data_model = get_daily_data_model_by_code(stock_code)
        print(f"📊 [单日计算] 获取模型: 因子={chip_factor_model}, 筹码={chips_model}, 日线={daily_data_model}")
        # 获取股票
        stock = await sync_to_async(StockInfo.objects.filter(stock_code=stock_code).first)()
        if not stock:
            print(f"❌ [单日计算] 股票不存在: {stock_code}")
            return {'status': 'failed', 'error': '股票不存在'}
        print(f"📊 [单日计算] 获取股票: {stock.stock_code} - {stock.stock_name}")
        # 检查是否已计算
        existing = await sync_to_async(chip_factor_model.objects.filter(stock=stock, trade_time=trade_date, calc_status='success').exists)()
        if existing:
            print(f"⚠️ [单日计算] 已计算过，跳过")
            return {'status': 'success', 'message': '已计算'}
        # 获取数据
        chip_perf = await sync_to_async(StockCyqPerf.objects.filter(stock=stock, trade_time=trade_date).first)()
        if not chip_perf:
            print(f"❌ [单日计算] 无筹码性能数据")
            return {'status': 'failed', 'error': '无筹码性能数据'}
        print(f"📊 [单日计算] 获取筹码性能数据成功")
        # 获取筹码分布
        chips_data = await sync_to_async(list)(chips_model.objects.filter(stock=stock, trade_time=trade_date).values('price', 'percent'))
        if not chips_data:
            print(f"❌ [单日计算] 无筹码分布数据")
            return {'status': 'failed', 'error': '无筹码分布数据'}
        chips_df = pd.DataFrame(chips_data)
        print(f"📊 [单日计算] 筹码分布数据: {len(chips_data)}条")
        # 获取日K线
        daily_kline = await sync_to_async(daily_data_model.objects.filter(stock=stock, trade_time=trade_date).first)()
        if not daily_kline:
            print(f"❌ [单日计算] 无日K线数据")
            return {'status': 'failed', 'error': '无日K线数据'}
        print(f"📊 [单日计算] 日K线数据: 收盘价={daily_kline.close_qfq}")
        # 获取前一日数据
        prev_date = trade_date - timedelta(days=1)
        prev_chips_data = await sync_to_async(list)(chips_model.objects.filter(stock=stock, trade_time=prev_date).values('price', 'percent'))
        prev_chips_df = pd.DataFrame(prev_chips_data) if prev_chips_data else pd.DataFrame()
        print(f"📊 [单日计算] 前一日筹码数据: {len(prev_chips_data) if prev_chips_data else 0}条")
        # 获取历史价格
        historical_prices = await get_historical_prices_for_stock(stock_code, trade_date, ChipTaskConfig.HISTORICAL_DAYS_FOR_MA)
        print(f"📊 [单日计算] 历史价格数据: {len(historical_prices)}条")
        # 获取历史因子
        historical_factors = await get_historical_chip_factors(chip_factor_model, stock, trade_date, 5)
        print(f"📊 [单日计算] 历史因子数据: {len(historical_factors)}条")
        # 准备数据
        chip_perf_dict = {'weight_avg': chip_perf.weight_avg, 'his_high': chip_perf.his_high, 'his_low': chip_perf.his_low, 'cost_5pct': chip_perf.cost_5pct, 'cost_15pct': chip_perf.cost_15pct, 'cost_50pct': chip_perf.cost_50pct, 'cost_85pct': chip_perf.cost_85pct, 'cost_95pct': chip_perf.cost_95pct, 'winner_rate': chip_perf.winner_rate}
        daily_kline_dict = {'close': daily_kline.close_qfq, 'open': daily_kline.open_qfq, 'high': daily_kline.high_qfq, 'low': daily_kline.low_qfq, 'vol': daily_kline.vol, 'amount': daily_kline.amount, 'pct_change': daily_kline.pct_change}
        # 计算基础因子（包含main_cost_range_ratio和high_position_lock_ratio_90的计算）
        factors = ChipFactorCalculator.calculate_complete_factors(chip_perf_data=chip_perf_dict, chip_dist_data=chips_df, daily_basic_data={}, daily_kline_data=daily_kline_dict, prev_chip_dist_data=prev_chips_df, historical_prices=historical_prices, historical_chip_factors=historical_factors)
        print(f"📊 [单日计算] 计算基础因子完成，字段数量：{len(factors)}")
        # 打印关键字段检查
        if 'main_cost_range_ratio' in factors:
            print(f"📊 [单日计算] main_cost_range_ratio: {factors['main_cost_range_ratio']:.4f}")
        else:
            print(f"⚠️ [单日计算] main_cost_range_ratio 未计算")
        if 'high_position_lock_ratio_90' in factors:
            print(f"📊 [单日计算] high_position_lock_ratio_90: {factors['high_position_lock_ratio_90']:.4f}")
        else:
            print(f"⚠️ [单日计算] high_position_lock_ratio_90 未计算")
        # 尝试从数据库加载持有时间矩阵因子
        try:
            from utils.model_helpers import get_chip_holding_matrix_model_by_code
            HoldingMatrixModel = get_chip_holding_matrix_model_by_code(stock_code)
            print(f"💾 [单日计算] 尝试加载持有矩阵模型: {HoldingMatrixModel}")
            holding_record = await sync_to_async(HoldingMatrixModel.objects.filter(stock=stock, trade_time=trade_date, calc_status='success').first)()
            if holding_record:
                print(f"✅ [单日计算] 找到已计算的持有矩阵记录，ID={holding_record.id}")
                # 直接使用数据库中的因子值
                factors['avg_holding_days'] = holding_record.avg_holding_days if holding_record.avg_holding_days is not None else 100.0
                factors['short_term_chip_ratio'] = holding_record.short_term_ratio if holding_record.short_term_ratio is not None else 0.2
                factors['long_term_chip_ratio'] = holding_record.long_term_ratio if holding_record.long_term_ratio is not None else 0.5
                # 计算中短线筹码（5-60日）
                short_term = factors.get('short_term_chip_ratio', 0.2)
                long_term = factors.get('long_term_chip_ratio', 0.5)
                factors['mid_term_ratio'] = max(0, 1.0 - short_term - long_term)
                print(f"📊 [单日计算] 从数据库加载持有时间因子成功")
            else:
                print(f"⚠️ [单日计算] 未找到已计算的持有矩阵，使用ChipFactorCalculator中的逻辑计算相关因子")
        except Exception as e:
            print(f"⚠️ [单日计算] 加载持有矩阵因子失败：{e}，使用ChipFactorCalculator中的逻辑")
        # 保存筹码因子
        print(f"💾 [单日计算] 开始保存筹码因子")
        await save_chip_factors(chip_factor_model, stock, trade_date, factors)
        print(f"💾 [单日计算] 因子保存完成，总计字段数：{len(factors)}")
        # 验证保存的字段
        verify_result = await verify_chip_factor_saved(stock_code, trade_date)
        if verify_result.get('exists'):
            print(f"✅ [单日计算] 验证通过: {stock_code} {trade_date} 已保存到数据库")
            if 'field_main_cost_range_ratio' in verify_result:
                print(f"📊 [单日计算] 保存的main_cost_range_ratio: {verify_result['field_main_cost_range_ratio']}")
            if 'field_high_position_lock_ratio_90' in verify_result:
                print(f"📊 [单日计算] 保存的high_position_lock_ratio_90: {verify_result['field_high_position_lock_ratio_90']}")
        return {'status': 'success', 'message': '计算完成'}
    except Exception as e:
        logger.error(f"计算股票 {stock_code} 日期 {trade_date} 失败: {e}")
        print(f"❌ [单日计算异常] {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}

@celery_app.task(bind=True, name='tasks.chip_factor_tasks.calculate_holding_matrix_batch', queue=ChipTaskConfig.get_queue_name())
def calculate_holding_matrix_batch(self, stock_codes: List[str], start_date: str, end_date: str, market: str = None) -> Dict:
    """批量计算多只股票的持有时间矩阵（使用AdvancedChipDynamicsService）版本：重构适配AdvancedChipDynamicsService
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        market: 市场标识
    Returns:
        Dict: 计算结果
    """
    try:
        logger.info(f"开始批量计算持有时间矩阵，股票数量: {len(stock_codes)}")
        start_date_obj = parse_date(start_date)
        end_date_obj = parse_date(end_date)
        results = {'total': len(stock_codes), 'success': 0, 'failed': 0, 'details': []}
        print(f"📋 [持有矩阵批量] 开始处理 {len(stock_codes)} 只股票")
        print(f"📅 [持有矩阵批量] 日期范围: {start_date_obj} 到 {end_date_obj}")
        # 按股票顺序处理
        for stock_index, stock_code in enumerate(stock_codes):
            try:
                print(f"🔴 [持有矩阵单股] 开始处理第 {stock_index + 1}/{len(stock_codes)} 只股票: {stock_code}")
                # 计算单个股票的持有时间矩阵（使用新的AdvancedChipDynamicsService）
                result = calculate_single_stock_holding_matrix_sync(stock_code, start_date_obj, end_date_obj)
                if result.get('status') == 'success':
                    results['success'] += 1
                    processed_dates = result.get('processed_dates', 0)
                    print(f"✅ [持有矩阵单股完成] {stock_code} 处理完成，成功 {processed_dates} 个交易日")
                else:
                    results['failed'] += 1
                    error_msg = result.get('error', '未知错误')
                    print(f"❌ [持有矩阵单股失败] {stock_code} 处理失败: {error_msg}")
                results['details'].append({'stock_code': stock_code, 'stock_index': stock_index + 1, **result})
                # 每完成5只股票打印一次进度
                if (stock_index + 1) % 5 == 0:
                    print(f"📊 [持有矩阵进度] 已完成 {stock_index + 1}/{len(stock_codes)} 只股票")
                    print(f"📊 [持有矩阵进度] 成功: {results['success']}, 失败: {results['failed']}")
            except Exception as e:
                results['failed'] += 1
                print(f"❌ [持有矩阵单股异常] {stock_code} 处理异常: {e}")
                results['details'].append({'stock_code': stock_code, 'stock_index': stock_index + 1, 'status': 'error', 'error': str(e), 'processed_dates': 0})
        logger.info(f"持有时间矩阵批量计算完成: 成功 {results['success']}, 失败 {results['failed']}")
        print(f"✅ [持有矩阵批量完成] 总计: 成功 {results['success']}, 失败 {results['failed']}")
        return results
    except Exception as e:
        logger.error(f"批量计算持有时间矩阵失败: {e}", exc_info=True)
        print(f"❌ [持有矩阵批量异常] {e}")
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
    print(f"股票 {stock_code} 综合计算已调度，任务ID列表: {task_ids}")
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
        from stock_models.index import TradeCalendar
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

# ========== 工具函数 ==========
def schedule_comprehensive_calculation(
    start_date_str: str = None,
    end_date_str: str = None,
    market: str = None,
    include_chip_factors: bool = True,
    include_holding_matrix: bool = True
) -> List[str]:
    """
    调度综合计算（命令行调用），改为链式调度：先计算持有矩阵，再计算筹码因子
    Args:
        start_date_str: 开始日期
        end_date_str: 结束日期
        market: 市场标识
        include_chip_factors: 是否包含筹码因子计算
        include_holding_matrix: 是否包含持有矩阵计算
    Returns:
        List[str]: 任务ID列表
    """
    task_ids = []
    if start_date_str is None:
        start_date_str = ChipTaskConfig.DEFAULT_START_DATE
    if end_date_str is None:
        end_date_str = get_last_trade_date().strftime('%Y%m%d')
    print(f"🔗 [链式调度开始] 日期范围: {start_date_str} 到 {end_date_str}")
    # 导入 Celery chain
    from celery import chain
    # 1. 如果同时需要持有矩阵和筹码因子，创建链式任务
    if include_holding_matrix and include_chip_factors:
        print(f"🔗 [链式调度] 创建链式任务：持有矩阵 → 筹码因子")
        # 创建任务链：先执行持有矩阵计算，再执行筹码因子计算
        task_chain = chain(
            schedule_holding_matrix_calculation.s(
                stock_codes=None,
                start_date_str=start_date_str,
                end_date_str=end_date_str
            ),
            schedule_chip_factor_calculation.s(
                stock_codes=None,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
                batch_mode=True
            )
        )
        # 执行链式任务
        chain_result = task_chain.delay()
        task_ids.append(chain_result.id)
        print(f"✅ [链式调度] 链式任务已创建: ID={chain_result.id}")
        print(f"📋 [链式调度] 任务链: 持有矩阵计算 → 筹码因子计算")
    # 2. 如果只需要持有矩阵
    elif include_holding_matrix and not include_chip_factors:
        print(f"⏳ [链式调度] 仅调度持有矩阵计算任务...")
        holding_task = schedule_holding_matrix_calculation.delay(
            stock_codes=None,
            start_date_str=start_date_str,
            end_date_str=end_date_str
        )
        task_ids.append(holding_task.id)
        print(f"✅ [链式调度] 持有矩阵任务已调度: {holding_task.id}")
    # 3. 如果只需要筹码因子
    elif not include_holding_matrix and include_chip_factors:
        print(f"⏳ [链式调度] 仅调度筹码因子计算任务...")
        chip_task = schedule_chip_factor_calculation.delay(
            stock_codes=None,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            batch_mode=True
        )
        task_ids.append(chip_task.id)
        print(f"✅ [链式调度] 筹码因子任务已调度: {chip_task.id}")
    else:
        print(f"⚠️ [链式调度] 未选择任何计算任务，请设置 include_chip_factors 或 include_holding_matrix 为 True")
    print(f"✅ [链式调度完成] 任务ID列表: {task_ids}")
    return task_ids

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