# tasks/strategy_tasks.py

import asyncio  # 保留异步支持
import logging
from celery import shared_task  # 使用 shared_task 以确保兼容
from chaoyue_dreams.celery import app as celery_app  # 假设这是您的Celery app
import pandas as pd
from channels.layers import get_channel_layer
from channels.db import database_sync_to_async  # 用于同步 ORM 查询
from services.indicator_services import IndicatorService
from strategies.base import SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, SIGNAL_STRONG_BUY, SIGNAL_STRONG_SELL
from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollEnhancedStrategy
from utils.cache_set import StrategyCacheSet
from dao_manager.daos.stock_basic_dao import StockBasicDAO  # 假设主任务需要

logger = logging.getLogger('tasks')  # 或者你使用的 logger 名称

@database_sync_to_async  # 将同步的 ORM 查询包装成异步
def _get_favorited_user_ids(self, stock_id: int) -> list[int]:
    """根据股票 ID 获取关注该股票的所有用户 ID 列表"""
    from users.models import FavoriteStock
    return list(FavoriteStock.objects.filter(stock_id=stock_id).values_list('user_id', flat=True))

@celery_app.task(bind=True, name='tasks.strategy.run_strategy_for_single_stock_task')
async def run_strategy_for_single_stock_task(self, stock_code: str):
    """
    为单支股票执行 MACD+RSI+KDJ+BOLL 策略计算并缓存结果。
    这是实际执行策略计算的 Celery Worker 任务。
    """
    log_prefix = f"[{stock_code}][StrategyTask]"
    # logger.info(f"{log_prefix} 开始执行 MACD+RSI+KDJ+BOLL 策略计算任务")
    service = None
    strategy = None
    cache_setter = None
    merged_data = None
    main_timeframe = '15'  # 或者从配置/策略对象获取
    try:
        # 1. 实例化所需对象
        service = IndicatorService()
        strategy = MacdRsiKdjBollEnhancedStrategy()  # 假设这个策略的时间框架和参数是固定的
        cache_setter = StrategyCacheSet()
        if not all([service, strategy, cache_setter]):
            logger.error(f"{log_prefix} 某个核心对象未能成功实例化，任务终止。")
            return {'status': 'failed', 'error': 'Core object instantiation failed'}  # 返回错误字典
        # 2. 准备数据
        logger.debug(f"{log_prefix} 准备策略数据...")
        merged_data = await service.prepare_strategy_dataframe(
            stock_code=stock_code,
            timeframes=strategy.timeframes,
            strategy_params=strategy.params,
            limit_per_tf=1500  # 保持与原逻辑一致
        )
        # 3. 处理数据准备失败的情况
        if merged_data is None or merged_data.empty:
            logger.warning(f"{log_prefix} 未能准备策略所需数据，策略无法运行。将缓存空信号状态。")
            try:
                latest_timestamp_ref = pd.Timestamp.now(tz='UTC')  # 使用 UTC 时间戳
                signal_data_to_cache = {
                    'stock_code': stock_code, 'strategy_name': strategy.strategy_name,
                    'time_level': main_timeframe, 'timestamp': latest_timestamp_ref.isoformat(),
                    'signal': None, 'signal_display': 'No Data', 'score': None,  # 假设没有 score
                    'generated_at': pd.Timestamp.now(tz='UTC').isoformat()
                }
                # 使用 await 调用异步缓存方法
                cache_success = await cache_setter.macd_rsi_kdj_boll_data(
                    stock_code=stock_code, time_level=main_timeframe, data_to_cache=signal_data_to_cache
                )
                if cache_success:
                    logger.info(f"{log_prefix} 数据准备失败，已缓存空信号状态。")
                else:
                    logger.warning(f"{log_prefix} 缓存空信号状态失败。")
            except Exception as cache_err:
                logger.error(f"{log_prefix} 缓存空信号状态时发生错误: {cache_err}", exc_info=False)
            return {'status': 'success', 'message': 'Data preparation failed, cached empty signal'}  # 返回结果
        # 4. 运行策略 (策略的 run 方法本身是同步的，但我们在这里处理)
        logger.debug(f"{log_prefix} 运行策略计算...")
        signal_series = strategy.run(merged_data)  # 假设 run 是同步的
        # 5. 处理和解析信号结果
        latest_signal = None
        latest_timestamp = None
        signal_display = "No Signal"  # 默认值
        if signal_series is not None and not signal_series.empty:
            valid_signals = signal_series.dropna()
            if not valid_signals.empty:
                latest_signal = valid_signals.iloc[-1]
                latest_timestamp = valid_signals.index[-1]
                signal_map = {
                    SIGNAL_STRONG_BUY: "Strong Buy",
                    SIGNAL_BUY: "Buy",
                    SIGNAL_HOLD: "Hold",
                    SIGNAL_SELL: "Sell",
                    SIGNAL_STRONG_SELL: "Strong Sell",
                }
                signal_display = signal_map.get(latest_signal, "Unknown Signal Value")
                if signal_display == 'Hold':
                    logger.info(f"{log_prefix} 策略信号 @ {latest_timestamp}: {latest_signal} ({signal_display})")
                else:
                    logger.warning(f"{log_prefix} 策略信号 @ {latest_timestamp}: {latest_signal} ({signal_display})")
            else:
                # logger.info(f"{log_prefix} 策略运行完成，但所有信号均为 NaN。")
                signal_display = "No Signal (NaN)"
                if not merged_data.empty:
                    latest_timestamp = merged_data.index[-1]
                else:
                    latest_timestamp = pd.Timestamp.now(tz='UTC')
        else:
            # logger.info(f"{log_prefix} 策略运行完成，但未生成有效信号序列。")
            signal_display = "No Signal (Empty Series)"
            if not merged_data.empty:
                latest_timestamp = merged_data.index[-1]
            else:
                latest_timestamp = pd.Timestamp.now(tz='UTC')
        # 6. 缓存策略结果
        logger.debug(f"{log_prefix} 准备缓存策略结果...")
        try:
            if latest_timestamp and latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.tz_localize('Asia/Shanghai').tz_convert('UTC')
            elif latest_timestamp is None:
                latest_timestamp = pd.Timestamp.now(tz='UTC')
            signal_data_to_cache = {
                'stock_code': stock_code,
                'strategy_name': strategy.strategy_name,
                'time_level': main_timeframe,
                'timestamp': latest_timestamp.isoformat(),
                'signal': int(latest_signal) if latest_signal is not None and not pd.isna(latest_signal) else None,
                'signal_display': signal_display,
                'generated_at': pd.Timestamp.now(tz='UTC').isoformat()
            }
            cache_success = await cache_setter.macd_rsi_kdj_boll_data(
                stock_code=stock_code, time_level=main_timeframe, data_to_cache=signal_data_to_cache
            )
            if cache_success:
                # logger.debug(f"{log_prefix} 策略结果/状态成功缓存到 Redis。")
                if signal_data_to_cache.get('signal_display') == 'Hold':
                    pass
                else:
                    # 1. 获取关注该股票的所有用户 ID 列表
                    # 2. 将信号数据推送给所有关注该股票的用户
                    pass
            else:
                logger.warning(f"{log_prefix} 缓存策略结果/状态到 Redis 失败。")
        except Exception as cache_err:
            logger.error(f"{log_prefix} 缓存策略结果/状态时发生错误: {cache_err}", exc_info=True)
    except Exception as e:
        logger.error(f"{log_prefix} 处理策略时发生严重错误: {e}", exc_info=True)
        return {'status': 'failed', 'error': str(e)}
    finally:
        # logger.info(f"{log_prefix} 策略计算任务执行完毕。")
        return {'status': 'success'}  # 返回成功状态

# ... (文件末尾的其他任务定义) ...
