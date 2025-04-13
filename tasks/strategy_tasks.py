# tasks/strategy_tasks.py

import asyncio  # 保留异步支持
import logging
from celery import shared_task  # 使用 shared_task 以确保兼容
from chaoyue_dreams.celery import app as celery_app  # 假设这是您的Celery app
import pandas as pd
from services.indicator_services import IndicatorService
from strategies.base import SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, SIGNAL_STRONG_BUY, SIGNAL_STRONG_SELL
from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollEnhancedStrategy
from utils.cache_set import StrategyCacheSet
from dao_manager.daos.stock_basic_dao import StockBasicDAO # 假设主任务需要

logger = logging.getLogger('strategy') # 或者你使用的 logger 名称

@celery_app.task(bind=True, name='tasks.strategy.run_strategy_for_single_stock_task')
def run_strategy_for_single_stock_task(self, stock_code: str):
    """
    为单支股票执行 MACD+RSI+KDJ+BOLL 策略计算并缓存结果。
    这是实际执行策略计算的 Celery Worker 任务。
    """
    async def inner_async_function(stock_code: str):
        log_prefix = f"[{stock_code}][StrategyTask]"
        logger.info(f"{log_prefix} 开始执行 MACD+RSI+KDJ+BOLL 策略计算任务")
        service = None
        strategy = None
        cache_setter = None
        merged_data = None
        main_timeframe = '15' # 或者从配置/策略对象获取
        try:
            # 1. 实例化所需对象
            # 注意：确保这些类的实例化在高并发下是安全的，或者考虑依赖注入
            service = IndicatorService()
            strategy = MacdRsiKdjBollEnhancedStrategy() # 假设这个策略的时间框架和参数是固定的
            cache_setter = StrategyCacheSet()
            if not all([service, strategy, cache_setter]):
                logger.error(f"{log_prefix} 某个核心对象未能成功实例化，任务终止。")
                # 可以选择性地抛出异常让 Celery 重试，或者直接返回
                # raise self.retry(exc=InstantiateError("Core object instantiation failed"), countdown=60)
                return # 直接终止
            # 2. 准备数据
            logger.debug(f"{log_prefix} 准备策略数据...")
            merged_data = await service.prepare_strategy_dataframe(
                stock_code=stock_code,
                timeframes=strategy.timeframes,
                strategy_params=strategy.params,
                limit_per_tf=1500 # 保持与原逻辑一致
            )
            # 3. 处理数据准备失败的情况
            if merged_data is None or merged_data.empty:
                logger.warning(f"{log_prefix} 未能准备策略所需数据，策略无法运行。将缓存空信号状态。")
                try:
                    latest_timestamp_ref = pd.Timestamp.now(tz='UTC') # 使用 UTC 时间戳
                    signal_data_to_cache = {
                        'stock_code': stock_code, 'strategy_name': strategy.strategy_name,
                        'time_level': main_timeframe, 'timestamp': latest_timestamp_ref.isoformat(),
                        'signal': None, 'signal_display': 'No Data', 'score': None, # 假设没有 score
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
                    # 记录详细错误，但不影响任务完成状态（除非需要重试）
                    logger.error(f"{log_prefix} 缓存空信号状态时发生错误: {cache_err}", exc_info=False)
                return # 数据准备失败，任务结束
            # 4. 运行策略 (策略的 run 方法本身是同步的)
            logger.debug(f"{log_prefix} 运行策略计算...")
            signal_series = strategy.run(merged_data)
            # 5. 处理和解析信号结果
            latest_signal = None
            latest_timestamp = None
            signal_display = "No Signal" # 默认值
            if signal_series is not None and not signal_series.empty:
                # 移除 NaN 值再获取最后一个有效信号
                valid_signals = signal_series.dropna()
                if not valid_signals.empty:
                    latest_signal = valid_signals.iloc[-1]
                    latest_timestamp = valid_signals.index[-1] # 获取信号对应的时间戳
                    # 定义信号映射关系
                    signal_map = {
                        SIGNAL_STRONG_BUY: "Strong Buy",
                        SIGNAL_BUY: "Buy",
                        SIGNAL_HOLD: "Hold",
                        SIGNAL_SELL: "Sell",
                        SIGNAL_STRONG_SELL: "Strong Sell",
                    }
                    signal_display = signal_map.get(latest_signal, "Unknown Signal Value") # 处理未定义信号值
                    logger.info(f"{log_prefix} 策略信号 @ {latest_timestamp}: {latest_signal} ({signal_display})")
                else:
                    # 策略运行了，但所有结果都是 NaN
                    logger.info(f"{log_prefix} 策略运行完成，但所有信号均为 NaN。")
                    signal_display = "No Signal (NaN)"
                    # 即使没有有效信号，也记录最后的数据时间戳
                    if not merged_data.empty:
                        latest_timestamp = merged_data.index[-1]
                    else: # 理论上不会到这里，因为前面检查过 merged_data
                        latest_timestamp = pd.Timestamp.now(tz='UTC')
            else:
                # 策略没有返回 Series 或者返回了空 Series
                logger.info(f"{log_prefix} 策略运行完成，但未生成有效信号序列。")
                signal_display = "No Signal (Empty Series)"
                # 记录最后的数据时间戳
                if not merged_data.empty:
                    latest_timestamp = merged_data.index[-1]
                else:
                    latest_timestamp = pd.Timestamp.now(tz='UTC')
            # 6. 缓存策略结果 (使用 await)
            logger.debug(f"{log_prefix} 准备缓存策略结果...")
            try:
                # 确保时间戳是 timezone-aware (UTC)
                if latest_timestamp and latest_timestamp.tzinfo is None:
                    latest_timestamp = latest_timestamp.tz_localize('Asia/Shanghai').tz_convert('UTC') # 或者直接使用 UTC
                elif latest_timestamp is None:
                    latest_timestamp = pd.Timestamp.now(tz='UTC')
                signal_data_to_cache = {
                    'stock_code': stock_code,
                    'strategy_name': strategy.strategy_name,
                    'time_level': main_timeframe,
                    'timestamp': latest_timestamp.isoformat(), # ISO 格式字符串
                    # 确保信号是 Python 内置 int 或 None
                    'signal': int(latest_signal) if latest_signal is not None and not pd.isna(latest_signal) else None,
                    'signal_display': signal_display,
                    # 'score': score, # 如果有评分逻辑，在这里添加
                    'generated_at': pd.Timestamp.now(tz='UTC').isoformat() # 记录生成时间
                }
                # 使用 await 调用异步缓存方法
                cache_success = await cache_setter.macd_rsi_kdj_boll_data(
                    stock_code=stock_code, time_level=main_timeframe, data_to_cache=signal_data_to_cache
                )
                if cache_success:
                    logger.debug(f"{log_prefix} 策略结果/状态成功缓存到 Redis。")
                else:
                    logger.warning(f"{log_prefix} 缓存策略结果/状态到 Redis 失败。")
            except Exception as cache_err:
                logger.error(f"{log_prefix} 缓存策略结果/状态时发生错误: {cache_err}", exc_info=True)
        except Exception as e:
            # 捕获任务执行期间的任何未预料错误
            logger.error(f"{log_prefix} 处理策略时发生严重错误: {e}", exc_info=True)
        finally:
            # 清理逻辑（如果需要的话），例如关闭 service 或 cache_setter 中的连接（如果它们不是全局/共享的）
            # 通常 Celery Task 不需要手动关闭这些，除非它们管理着需要显式关闭的资源
            logger.info(f"{log_prefix} 策略计算任务执行完毕。")
    # 在同步任务中运行内部异步函数
    try:
        result = asyncio.run(inner_async_function(stock_code))
        return result  # 返回字典，便于序列化
    except Exception as e:
        logger.error(f"[{stock_code}][StrategyTask] 运行异步逻辑时发生错误: {e}", exc_info=True)
        return {'status': 'failed', 'error': str(e)}

# ... (文件末尾的其他任务定义) ...
